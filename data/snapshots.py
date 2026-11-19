"""Weekly data snapshots: validation, manifest, atomic activation, restricted Google Drive source.

Production CSV exports are sensitive and must never live in the Git repository. The app instead
reads a *validated weekly snapshot* from a data directory:

    <Drive root>/Current/                 <- the active snapshot (CSVs + manifest.json)
    <Drive root>/Archive/YYYY-MM-DD/      <- previous snapshots, same layout

Resolution order for the data directory (``resolve_data_dir``):

1. Restricted Google Drive (``gdrive`` secrets / ``GDRIVE_*`` env): ``Current`` is downloaded to a
   staging directory, validated, and only then atomically swapped into the local cache.
2. ``DASHBOARD_DATA_DIR`` - a local folder holding a snapshot (development, or a mounted volume).
3. The app folder itself (legacy behaviour; no manifest required).

A bad source never silently corrupts the others: validation reports per-file problems and the
dashboard KPIs each file feeds (``blocked_kpis``). See docs/data-governance.md.

The module is pure standard library (Drive access is a lazy, optional import) so it is unit-testable
without network or Streamlit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class SourceSpec:
    required: bool
    columns: Sequence[str]
    kpis: Sequence[str]


#: file -> what must be in it and which KPIs it feeds.
SOURCES: Dict[str, SourceSpec] = {
    "email.csv": SourceSpec(True, ["Status", "Date/Time Opened", "Completion Date"],
                            ["Email attainment", "Overall Service Attainment", "Email backlog",
                             "Operational FCR", "Email breakdowns"]),
    "report_items.csv": SourceSpec(True, ["Work Item: Name", "Service Channel: Developer Name", "Start DT"],
                                   ["Workload & efficiency", "Email queue breakdown",
                                    "Schedule adherence (handling)"]),
    "report_messaging_agent_work.csv": SourceSpec(
        True, ["Messaging Session Name", "Status", "Assign Date", "Accept Date"],
        ["Messaging attainment", "Overall Service Attainment", "Messaging diagnostics"]),
    "report_messaging_metrics.csv": SourceSpec(
        True, ["Messaging Session Name", "Messaging Session Metric Type", "Messaging Session Metric Value"],
        ["Messaging attainment", "Overall Service Attainment", "Messaging diagnostics"]),
    "report_presence.csv": SourceSpec(True, ["Service Presence Status: Developer Name", "Start DT", "End DT"],
                                      ["Schedule adherence", "Utilisation"]),
    "chat.csv": SourceSpec(True, ["Date/Time Opened"], ["Historical Live Chat"]),
    "shifts.csv": SourceSpec(False, [], ["Schedule adherence"]),
    "survey.csv": SourceSpec(False, ["Survey Taker: ID", "Response", "Survey Question: Question Title"],
                             ["CSAT", "Survey FCR", "NPS"]),
}


def _colkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header_and_rows(path: Path):
    """(header list, data-row count) trying utf-8-sig then cp1252."""
    last: Optional[Exception] = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, newline="", encoding=enc) as f:
                rdr = csv.reader(f)
                header = next(rdr, [])
                n = sum(1 for _ in rdr)
            return [h.strip() for h in header], n
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # pragma: no cover


# --------------------------------------------------------------------------- manifest
def build_manifest(directory: Path, snapshot_date: Optional[date] = None) -> dict:
    """Describe every known source file present in ``directory`` (sha256, bytes, rows)."""
    directory = Path(directory)
    files = {}
    for name in SOURCES:
        p = directory / name
        if p.is_file():
            _, rows = _read_header_and_rows(p)
            files[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "rows": rows}
    return {"snapshot_date": (snapshot_date or date.today()).isoformat(),
            "created": datetime.now().isoformat(timespec="seconds"), "files": files}


def write_manifest(directory: Path, snapshot_date: Optional[date] = None) -> Path:
    path = Path(directory) / MANIFEST_NAME
    path.write_text(json.dumps(build_manifest(directory, snapshot_date), indent=2), encoding="utf-8")
    return path


def read_manifest(directory: Path) -> Optional[dict]:
    p = Path(directory) / MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"_unreadable": True}


# --------------------------------------------------------------------------- validation
@dataclass
class FileStatus:
    name: str
    present: bool
    ok: bool
    rows: int = 0
    problems: List[str] = field(default_factory=list)


@dataclass
class SnapshotReport:
    directory: str
    snapshot_date: Optional[str]
    manifest_present: bool
    files: Dict[str, FileStatus]
    problems: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when every REQUIRED source is present and valid."""
        return all(self.files[n].ok for n, s in SOURCES.items() if s.required)

    @property
    def blocked_kpis(self) -> List[str]:
        """KPIs fed by a source that is missing or failed validation (and only those)."""
        out: List[str] = []
        for name, spec in SOURCES.items():
            st = self.files.get(name)
            if st is not None and not st.ok and (spec.required or st.present):
                for k in spec.kpis:
                    if k not in out:
                        out.append(k)
        return out

    @property
    def failing_files(self) -> List[str]:
        return [n for n, s in self.files.items() if not s.ok and (SOURCES[n].required or s.present)]


def validate_snapshot(directory: Path, require_manifest: bool = False) -> SnapshotReport:
    directory = Path(directory)
    manifest = read_manifest(directory)
    problems: List[str] = []
    if manifest is None:
        if require_manifest:
            problems.append("manifest.json is missing: snapshot is unverified.")
        manifest_files: dict = {}
    elif manifest.get("_unreadable"):
        problems.append("manifest.json could not be read.")
        manifest_files = {}
    else:
        manifest_files = manifest.get("files", {})

    files: Dict[str, FileStatus] = {}
    for name, spec in SOURCES.items():
        p = _find_case_insensitive(directory, name)
        if p is None:
            files[name] = FileStatus(name, False, not spec.required,
                                     problems=[] if not spec.required else ["file is missing"])
            continue
        st = FileStatus(name, True, True)
        try:
            header, rows = _read_header_and_rows(p)
            st.rows = rows
            have = {_colkey(h) for h in header}
            missing = [c for c in spec.columns if _colkey(c) not in have]
            if missing:
                st.problems.append("missing column(s): " + ", ".join(missing))
            if rows == 0:
                st.problems.append("file has no data rows")
            entry = manifest_files.get(name)
            if entry is not None:
                if entry.get("sha256") and entry["sha256"] != sha256_file(p):
                    st.problems.append("checksum does not match the manifest (file changed or corrupt)")
                if entry.get("rows") is not None and entry["rows"] != rows:
                    st.problems.append(f"row count {rows} differs from the manifest ({entry['rows']})")
            elif manifest is not None and not manifest.get("_unreadable"):
                st.problems.append("file is not listed in the manifest")
        except Exception as exc:  # unreadable / binary / permissions
            st.problems.append(f"could not be read: {type(exc).__name__}")
        st.ok = not st.problems
        files[name] = st
    return SnapshotReport(str(directory), (manifest or {}).get("snapshot_date"),
                          manifest is not None, files, problems)


def _find_case_insensitive(directory: Path, name: str) -> Optional[Path]:
    p = directory / name
    if p.is_file():
        return p
    if directory.is_dir():
        for child in directory.iterdir():
            if child.is_file() and child.name.lower() == name.lower():
                return child
    return None


# --------------------------------------------------------------------------- atomic activation
def activate_snapshot(staging: Path, current: Path) -> None:
    """Swap ``staging`` in as ``current``; the previous one is kept as ``<current>.prev``.

    Readers see either the complete old snapshot or the complete new one. On failure the old
    snapshot is restored.
    """
    staging, current = Path(staging), Path(current)
    prev = current.with_name(current.name + ".prev")
    if prev.exists():
        shutil.rmtree(prev)
    had_current = current.exists()
    if had_current:
        os.replace(current, prev)
    try:
        os.replace(staging, current)
    except Exception:
        if had_current:
            os.replace(prev, current)
        raise


# --------------------------------------------------------------------------- Google Drive
class DriveClient(Protocol):
    def list_children(self, folder_id: str) -> List[dict]: ...      # [{id, name, mimeType}]
    def download(self, file_id: str, dest: Path) -> None: ...


FOLDER_MIME = "application/vnd.google-apps.folder"


class GoogleDriveClient:  # pragma: no cover - needs network and credentials
    """Read-only Drive client using a service account (lazy optional imports)."""

    def __init__(self, service_account_info: Mapping):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_info(
            dict(service_account_info), scopes=["https://www.googleapis.com/auth/drive.readonly"])
        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    def list_children(self, folder_id: str) -> List[dict]:
        out, token = [], None
        while True:
            r = self._svc.files().list(
                q=f"'{folder_id}' in parents and trashed = false", pageSize=200, pageToken=token,
                fields="nextPageToken, files(id, name, mimeType)",
                supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            out += r.get("files", [])
            token = r.get("nextPageToken")
            if not token:
                return out

    def download(self, file_id: str, dest: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload
        req = self._svc.files().get_media(fileId=file_id, supportsAllDrives=True)
        with open(dest, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()


def download_current(client: DriveClient, root_folder_id: str, dest: Path) -> List[str]:
    """Download every file of ``<root>/Current`` into ``dest``. Returns the file names."""
    kids = client.list_children(root_folder_id)
    cur = next((k for k in kids if k["name"] == "Current" and k.get("mimeType") == FOLDER_MIME), None)
    if cur is None:
        raise FileNotFoundError("No 'Current' folder found in the Drive data folder.")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for f in client.list_children(cur["id"]):
        if f.get("mimeType") == FOLDER_MIME:
            continue
        name = Path(f["name"]).name           # never let a Drive name escape the staging dir
        client.download(f["id"], dest / name)
        names.append(name)
    return names


# --------------------------------------------------------------------------- resolution
@dataclass
class DataSource:
    path: Path
    mode: str                 # "drive" | "local-dir" | "app-folder"
    report: SnapshotReport
    message: str = ""


def _gdrive_config(secrets: Optional[Mapping], env: Mapping) -> Optional[dict]:
    g = (secrets or {}).get("gdrive")
    if g and g.get("folder_id") and g.get("service_account"):
        sa = g["service_account"]
        return {"folder_id": g["folder_id"], "service_account": dict(sa) if not isinstance(sa, str) else json.loads(sa)}
    if env.get("GDRIVE_FOLDER_ID") and env.get("GDRIVE_SERVICE_ACCOUNT_JSON"):
        return {"folder_id": env["GDRIVE_FOLDER_ID"],
                "service_account": json.loads(env["GDRIVE_SERVICE_ACCOUNT_JSON"])}
    return None


def resolve_data_dir(app_dir: Path, secrets: Optional[Mapping] = None, env: Optional[Mapping] = None,
                     client: Optional[DriveClient] = None) -> DataSource:
    env = os.environ if env is None else env
    cfg = _gdrive_config(secrets, env)
    if cfg is not None or client is not None:
        cache_root = Path(env.get("DASHBOARD_CACHE_DIR") or Path(tempfile.gettempdir()) / "dept_dashboard")
        cache_root.mkdir(parents=True, exist_ok=True)
        current = cache_root / "current"
        try:
            c = client or GoogleDriveClient(cfg["service_account"])
            folder = cfg["folder_id"] if cfg else "root"
            staging = Path(tempfile.mkdtemp(prefix="staging_", dir=cache_root))
            download_current(c, folder, staging)
            report = validate_snapshot(staging, require_manifest=True)
            if not report.usable:
                shutil.rmtree(staging, ignore_errors=True)
                if current.exists():   # keep serving the last good snapshot
                    return DataSource(current, "drive", validate_snapshot(current, True),
                                      "New Drive snapshot failed validation; showing the previous "
                                      "valid snapshot. Problems: " + "; ".join(
                                          f"{n}: {', '.join(s.problems)}" for n, s in report.files.items() if s.problems))
                return DataSource(staging, "drive", report, "Drive snapshot failed validation.")
            activate_snapshot(staging, current)
            return DataSource(current, "drive", validate_snapshot(current, True))
        except Exception as exc:
            if current.exists():
                return DataSource(current, "drive", validate_snapshot(current, True),
                                  f"Drive refresh failed ({type(exc).__name__}); showing the last good snapshot.")
            return DataSource(cache_root / "empty", "drive",
                              validate_snapshot(cache_root / "empty"),
                              f"Drive snapshot could not be loaded ({type(exc).__name__}).")
    if env.get("DASHBOARD_DATA_DIR"):
        d = Path(env["DASHBOARD_DATA_DIR"])
        return DataSource(d, "local-dir", validate_snapshot(d))
    return DataSource(Path(app_dir), "app-folder", validate_snapshot(Path(app_dir)))


# --------------------------------------------------------------------------- source cutoffs
@dataclass
class CutoffReport:
    common: Optional[object]         # latest timestamp every source covers (min of horizons)
    latest: Optional[object]
    mismatched: List[str]            # sources that stop more than ``tolerance`` before the latest


def cutoff_report(horizons: Mapping[str, Optional[object]], tolerance_hours: float = 24.0) -> CutoffReport:
    """``horizons``: source -> its latest timestamp (None = no data). Never mix mismatched cutoffs."""
    import pandas as pd
    h = {k: pd.Timestamp(v) for k, v in horizons.items() if v is not None and not pd.isna(v)}
    if not h:
        return CutoffReport(None, None, [])
    latest, common = max(h.values()), min(h.values())
    tol = pd.Timedelta(hours=tolerance_hours)
    return CutoffReport(common, latest, sorted(k for k, v in h.items() if latest - v > tol))


# --------------------------------------------------------------------------- CLI
def _main(argv: Sequence[str]) -> int:  # pragma: no cover - thin wrapper
    import argparse
    ap = argparse.ArgumentParser(prog="python -m data.snapshots")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-manifest", help="write manifest.json for a snapshot folder")
    b.add_argument("directory"); b.add_argument("--date", default=None)
    v = sub.add_parser("validate", help="validate a snapshot folder")
    v.add_argument("directory"); v.add_argument("--require-manifest", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build-manifest":
        d = date.fromisoformat(a.date) if a.date else None
        print(write_manifest(Path(a.directory), d))
        return 0
    rep = validate_snapshot(Path(a.directory), a.require_manifest)
    for n, s in rep.files.items():
        print(f"{'OK ' if s.ok else 'BAD'} {n}: {s.rows} rows {'; '.join(s.problems)}")
    print("usable:", rep.usable, "| blocked KPIs:", ", ".join(rep.blocked_kpis) or "none")
    return 0 if rep.usable else 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv[1:]))
