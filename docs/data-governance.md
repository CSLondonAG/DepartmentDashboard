# Data governance: storage, snapshots and Git hygiene (Change 7)

Production exports (email, work items, Messaging, presence, shifts, surveys, chat) contain customer
and staff personal data. **They must never be committed to Git.** The repository holds code and
*synthetic* fixtures only (`tests/fixtures/`, `tests/fixtures.py`).

## Weekly snapshot layout (restricted Google Drive)

```
<Drive data folder>/
  Current/                 <- the active snapshot the dashboard reads
    email.csv  report_items.csv  report_messaging_agent_work.csv  report_messaging_metrics.csv
    report_presence.csv  chat.csv  shifts.csv  survey.csv  manifest.json
  Archive/
    2026-11-16/            <- previous weekly snapshots, same layout
```

Weekly runbook (Monday, after the Sunday 23:59 CAT cutoff):

1. Put the new exports in an empty local folder.
2. `python -m data.snapshots build-manifest <folder> --date YYYY-MM-DD` (records sha256, bytes, rows).
3. `python -m data.snapshots validate <folder> --require-manifest` (must print `usable: True`).
4. In Drive: move the existing `Current/*` into `Archive/<previous date>/`, then upload the new
   files and `manifest.json` into `Current/`.

The app downloads `Current/` to a staging folder, validates it (required files, required columns,
non-empty, checksums and row counts against the manifest) and only then swaps it in atomically.
If a new snapshot fails validation, the **last good snapshot keeps being served** and the failing
files and the KPIs they feed are listed in the "Data files & quality" panel. Weekly source dates
are shown there too, with a warning when sources end on different dates.

## Configuration

Streamlit secrets (`.streamlit/secrets.toml`, never committed) or environment variables:

```toml
[gdrive]
folder_id = "<id of the Drive data folder>"
service_account = """{ ...service account JSON... }"""   # read-only on that folder only
```

or `GDRIVE_FOLDER_ID` + `GDRIVE_SERVICE_ACCOUNT_JSON`. Share the Drive folder with the service
account (Viewer). Without Drive config, set `DASHBOARD_DATA_DIR=<folder with a snapshot>`; with
neither, the app looks in its own folder (legacy local use).

> Open decision (spec 5.7): the Drive auth mechanism. A read-only service account is implemented as
> the default; swap `GoogleDriveClient` in `data/snapshots.py` if another mechanism is chosen.

## Removing production data from the repository **history** (requires a repo admin)

This change stops tracking the CSVs and ignores them going forward, but earlier commits still
contain them. Rewriting history needs a force-push to a shared repository, so it is **not done by
this change**. To finish it:

1. Ask all collaborators to push or abandon open work (their clones will need re-cloning).
2. On a fresh clone: `git filter-repo --invert-paths --path-glob '*.csv' --path chat_transcripts.csv`
   (then re-add `tests/fixtures/*.csv` from the current tree, or use `--path-glob` exclusions).
3. Force-push all branches and tags; ask GitHub Support to purge cached views and PR refs
   (`refs/pull/*`) if the data is sensitive.
4. Treat the data as exposed: rotate any credentials found, and review whether notification
   obligations apply.

Until then, anyone with access to the repository can still read the old exports from history.

## Known limitations

* A source that fails validation is *reported with the KPIs it feeds* (and the last good snapshot is
  kept), but the dashboard still stops if a required file is entirely absent; per-KPI degradation of a
  single bad file inside one run is not implemented.
* Drive download is implemented against the Drive v3 API but has only been unit-tested with a fake
  client (no network in CI).
