import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.snapshots import (
    FOLDER_MIME, MANIFEST_NAME, SOURCES, activate_snapshot, build_manifest, cutoff_report,
    download_current, resolve_data_dir, validate_snapshot, write_manifest,
)

HEADERS = {
    "email.csv": "Status,Date/Time Opened,Completion Date,Case Number",
    "report_items.csv": "Work Item: Name,Service Channel: Developer Name,Start DT",
    "report_messaging_agent_work.csv": "Messaging Session Name,Status,Assign Date,Accept Date",
    "report_messaging_metrics.csv": "Messaging Session Name,Messaging Session Metric Type,Messaging Session Metric Value",
    "report_presence.csv": "Service Presence Status: Developer Name,Start DT,End DT",
    "chat.csv": "Date/Time Opened,Wait Time",
}


def make_snapshot(d: Path, skip=(), extra=None):
    d.mkdir(parents=True, exist_ok=True)
    for name, header in HEADERS.items():
        if name in skip:
            continue
        cols = len(header.split(","))
        (d / name).write_text(header + "\n" + ",".join(["x"] * cols) + "\n", encoding="utf-8")
    for name, text in (extra or {}).items():
        (d / name).write_text(text, encoding="utf-8")
    return d


# ------------------------------------------------------------------ validation
def test_valid_snapshot_is_usable_with_no_blocked_kpis(tmp_path):
    d = make_snapshot(tmp_path / "s")
    write_manifest(d, date(2026, 11, 23))
    rep = validate_snapshot(d, require_manifest=True)
    assert rep.usable and rep.blocked_kpis == [] and rep.snapshot_date == "2026-11-23"
    assert rep.files["shifts.csv"].ok and not rep.files["shifts.csv"].present   # optional


def test_manifest_records_hash_bytes_and_rows(tmp_path):
    d = make_snapshot(tmp_path / "s")
    m = build_manifest(d, date(2026, 11, 23))
    e = m["files"]["email.csv"]
    assert e["rows"] == 1 and e["bytes"] > 0 and len(e["sha256"]) == 64
    assert set(m["files"]) == set(HEADERS)


def test_missing_required_file_blocks_only_its_kpis(tmp_path):
    d = make_snapshot(tmp_path / "s", skip=("report_presence.csv",))
    rep = validate_snapshot(d)
    assert not rep.usable and rep.failing_files == ["report_presence.csv"]
    assert "Schedule adherence" in rep.blocked_kpis and "Utilisation" in rep.blocked_kpis
    assert "Email attainment" not in rep.blocked_kpis and "CSAT" not in rep.blocked_kpis


def test_missing_column_empty_file_and_checksum_mismatch(tmp_path):
    d = make_snapshot(tmp_path / "s", extra={"email.csv": "Status,Case Number\nOpen,1\n"})
    rep = validate_snapshot(d)
    assert "missing column(s): Date/Time Opened, Completion Date" in rep.files["email.csv"].problems[0]
    assert "Email attainment" in rep.blocked_kpis and "Workload & efficiency" not in rep.blocked_kpis

    d2 = make_snapshot(tmp_path / "t")
    write_manifest(d2)
    (d2 / "report_items.csv").write_text(HEADERS["report_items.csv"] + "\n", encoding="utf-8")   # no rows
    rep2 = validate_snapshot(d2, require_manifest=True)
    probs = " ".join(rep2.files["report_items.csv"].problems)
    assert "no data rows" in probs and "checksum" in probs
    assert not rep2.usable


def test_tampered_file_is_detected_by_the_manifest(tmp_path):
    d = make_snapshot(tmp_path / "s")
    write_manifest(d)
    (d / "chat.csv").write_text("Date/Time Opened,Wait Time\n1,2\n3,4\n", encoding="utf-8")
    rep = validate_snapshot(d, require_manifest=True)
    assert not rep.files["chat.csv"].ok and "Historical Live Chat" in rep.blocked_kpis
    assert rep.files["email.csv"].ok


def test_unlisted_file_and_missing_manifest_policy(tmp_path):
    d = make_snapshot(tmp_path / "s")
    assert validate_snapshot(d).usable                        # legacy folder: manifest optional
    rep = validate_snapshot(d, require_manifest=True)
    assert rep.problems and not rep.manifest_present
    write_manifest(d)
    (d / "survey.csv").write_text("Survey Taker: ID,Response,Survey Question: Question Title\n1,5,q\n",
                                  encoding="utf-8")
    assert "not listed in the manifest" in " ".join(validate_snapshot(d).files["survey.csv"].problems)


def test_column_names_match_case_and_punctuation_insensitively(tmp_path):
    d = make_snapshot(tmp_path / "s", extra={"email.csv": "status,DATE TIME OPENED,completion date\nOpen,a,b\n"})
    assert validate_snapshot(d).files["email.csv"].ok


def test_cp1252_files_are_readable(tmp_path):
    d = make_snapshot(tmp_path / "s")
    (d / "email.csv").write_bytes("Status,Date/Time Opened,Completion Date,Country\nOpen,a,b,C\xf4te\n".encode("cp1252"))
    assert validate_snapshot(d).files["email.csv"].ok


# ------------------------------------------------------------------ atomic activation
def test_activate_swaps_atomically_and_keeps_previous(tmp_path):
    old = make_snapshot(tmp_path / "current")
    (old / "marker.txt").write_text("old")
    new = make_snapshot(tmp_path / "staging")
    (new / "marker.txt").write_text("new")
    activate_snapshot(new, old)
    assert (tmp_path / "current" / "marker.txt").read_text() == "new"
    assert (tmp_path / "current.prev" / "marker.txt").read_text() == "old"
    assert not (tmp_path / "staging").exists()


def test_activate_restores_previous_on_failure(tmp_path):
    cur = make_snapshot(tmp_path / "current")
    (cur / "marker.txt").write_text("old")
    with pytest.raises(OSError):
        activate_snapshot(tmp_path / "does-not-exist", cur)
    assert (tmp_path / "current" / "marker.txt").read_text() == "old"


def test_activate_with_no_existing_current(tmp_path):
    new = make_snapshot(tmp_path / "staging")
    activate_snapshot(new, tmp_path / "current")
    assert (tmp_path / "current" / "email.csv").exists()


# ------------------------------------------------------------------ Drive (fake client)
class FakeDrive:
    def __init__(self, tree, contents):
        self.tree, self.contents = tree, contents

    def list_children(self, folder_id):
        return self.tree[folder_id]

    def download(self, file_id, dest):
        Path(dest).write_bytes(self.contents[file_id])


def drive_for(snapshot_dir: Path, bad=False):
    files, contents = [], {}
    for p in sorted(snapshot_dir.iterdir()):
        files.append({"id": p.name, "name": p.name, "mimeType": "text/csv"})
        contents[p.name] = p.read_bytes()
    if bad:
        contents["email.csv"] = b"tampered"
    tree = {"root": [{"id": "cur", "name": "Current", "mimeType": FOLDER_MIME},
                     {"id": "arc", "name": "Archive", "mimeType": FOLDER_MIME}],
            "cur": files, "arc": []}
    return FakeDrive(tree, contents)


def test_download_current_only_takes_the_current_folder_and_sanitises_names(tmp_path):
    snap = make_snapshot(tmp_path / "s")
    c = drive_for(snap)
    c.tree["cur"].append({"id": "evil", "name": "../../escape.csv", "mimeType": "text/csv"})
    c.contents["evil"] = b"x"
    names = download_current(c, "root", tmp_path / "dl")
    assert "escape.csv" in names and not (tmp_path / "escape.csv").exists()
    assert (tmp_path / "dl" / "email.csv").exists()
    with pytest.raises(FileNotFoundError):
        download_current(FakeDrive({"root": []}, {}), "root", tmp_path / "x")


def test_resolve_from_drive_validates_then_activates(tmp_path):
    snap = make_snapshot(tmp_path / "s")
    write_manifest(snap, date(2026, 11, 23))
    env = {"DASHBOARD_CACHE_DIR": str(tmp_path / "cache")}
    src = resolve_data_dir(tmp_path, env=env, client=drive_for(snap))
    assert src.mode == "drive" and src.report.usable and src.report.snapshot_date == "2026-11-23"
    assert (src.path / "email.csv").exists() and src.path.name == "current"


def test_bad_drive_snapshot_keeps_serving_the_last_good_one(tmp_path):
    snap = make_snapshot(tmp_path / "s")
    write_manifest(snap, date(2026, 11, 23))
    env = {"DASHBOARD_CACHE_DIR": str(tmp_path / "cache")}
    good = resolve_data_dir(tmp_path, env=env, client=drive_for(snap))
    assert good.report.usable
    bad = resolve_data_dir(tmp_path, env=env, client=drive_for(snap, bad=True))
    assert bad.path == good.path and bad.report.usable
    assert "failed validation" in bad.message and "checksum" in bad.message


def test_bad_first_drive_snapshot_is_reported_not_activated(tmp_path):
    snap = make_snapshot(tmp_path / "s")
    write_manifest(snap)
    env = {"DASHBOARD_CACHE_DIR": str(tmp_path / "cache")}
    src = resolve_data_dir(tmp_path, env=env, client=drive_for(snap, bad=True))
    assert not src.report.usable and not (tmp_path / "cache" / "current").exists()


def test_resolution_precedence_local_dir_then_app_folder(tmp_path):
    app = make_snapshot(tmp_path / "app")
    other = make_snapshot(tmp_path / "other")
    s = resolve_data_dir(app, env={"DASHBOARD_DATA_DIR": str(other)})
    assert s.mode == "local-dir" and s.path == other
    s = resolve_data_dir(app, env={})
    assert s.mode == "app-folder" and s.path == app


def test_cli_validate_exit_code(tmp_path):
    d = make_snapshot(tmp_path / "s")
    root = Path(__file__).resolve().parents[1]
    ok = subprocess.run(["python", "-m", "data.snapshots", "validate", str(d)], cwd=root,
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "usable: True" in ok.stdout
    (d / "email.csv").unlink()
    bad = subprocess.run(["python", "-m", "data.snapshots", "validate", str(d)], cwd=root,
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "Email attainment" in bad.stdout


# ------------------------------------------------------------------ cutoffs
def test_cutoff_report_flags_sources_that_stop_early():
    r = cutoff_report({"email": "2026-11-22 23:59", "presence": "2026-11-22 23:00",
                       "messaging": "2026-11-18 10:00", "survey": None})
    assert r.latest == pd.Timestamp("2026-11-22 23:59") and r.common == pd.Timestamp("2026-11-18 10:00")
    assert r.mismatched == ["messaging"]
    assert cutoff_report({}).common is None


def test_every_source_spec_has_kpis():
    assert all(s.kpis for s in SOURCES.values())


def test_no_production_csv_is_tracked_at_the_repo_root():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [f for f in out.stdout.splitlines() if f.lower().endswith(".csv")]
    assert all(f.startswith("tests/fixtures/") for f in tracked), tracked
    assert "/*.csv" in (root / ".gitignore").read_text()
