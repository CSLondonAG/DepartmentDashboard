# Department Dashboard Redesign — Build Spec

**Status:** Authoritative implementation spec. Freeze of all data-source decisions made after
investigating the real CSVs on 2026-11-19.
**Companion doc:** the approved brief `2026-11-19-department-performance-dashboard-redesign-design.md`
(Changes 1–7, acceptance criteria). Where the brief and this spec differ on a *data binding*, this
spec wins (it reflects the actual columns). Where they differ on *intent/targets*, the brief wins.
**Product:** `department_dashboard.py` (Streamlit, single file, ~3,900 lines).

> Purpose of this file: so no build session has to re-read the 573-line brief or re-profile 17k rows
> of CSV. All field decisions, formulas, thresholds, mappings and known limitations are here.

---

## 0. How to build (conventions — read first)

- **Scoped edits only.** Never rewrite `department_dashboard.py` wholesale. Locate with grep + ranged
  reads, change the minimum, edit by exact string.
- **Put new metric logic in new modules**, not inline in the monolith. Suggested:
  `metrics/service_attainment.py`, `metrics/adherence.py`, `metrics/backlog.py`, `metrics/breakdowns.py`,
  `data/ingest.py`. Import into `department_dashboard.py`. This keeps each change's context small and is
  lower-risk than a big-bang refactor of the existing file.
- **Tests per formula.** Every metric gets `tests/` coverage with hand-worked fixtures for the edge
  cases (SLA pass/fail/pending, abandonment, reopen, adherence cap, coverage gate, maturity). Validate
  by running `pytest`, not by eyeballing.
- **Every headline % must reconcile to a visible numerator and denominator.**
- **No agent-level output.** Agent names exist in the data (`User: Full Name`, `Case Owner`) — never
  display them or any per-agent table. Department aggregates only.
- **Provisional/coverage labelling is mandatory**, not optional polish (see §3, §4).
- Work the changes in order 1 → 7. Open a PR after **Change 1 + its tests** for review before
  continuing; then proceed.

---

## 1. Data sources and the NEW email export schema

Weekly CSVs (Windows-1252 / latin-1 encoded). The **email export schema changed on 2026-11-19**.

### 1.1 `email.csv` — NEW schema (adopt this)
Columns: `Status | Priority | Case Origin | Date/Time Opened | Completion Date | Re-opened |
Reopened Time | Elapsed Time (Hours) | Your Email Address | Case Number | Case Owner | Country`

> Note the older export also had `Milestone | Target Date | Target Response (Hours)`. The new one drops
> them. We do **not** need them — we recompute SLA ourselves (§2.1).

**Two ingestion quirks that MUST be handled:**
1. **Duplicated header row.** Row 2 repeats the column names (`Status == "Status"`). Drop any row whose
   `Status` value equals `"Status"`.
2. **Zero-padded `Case Number`** (e.g. `04282933`, previously `4282933`). Normalise the join/dedup key
   (strip leading zeros, or treat consistently as a string) so it matches `report_items.Work Item: Name`.

Timestamp format: `dd/mm/yyyy, HH:MM` (e.g. `22/10/2026, 16:20`). Dayfirst.

### 1.2 Other weekly files (unchanged)
- `report_items.csv` — agent work items. `Work Item: Name` (= case number), `Queue: Name`,
  `Assign/Accept/Close Date+Time`, `Handle Time`, `Start DT`, `End DT`. For touches, transfers, routing,
  email handling queue.
- `report_messaging_agent_work.csv` — `Chat Session ID`, `Status`, `Assign Date`, `Accept Date`,
  `Handle Time`, `Speed To Answer`, `Messaging Session Name`.
- `report_messaging_metrics.csv` — long format: `Messaging Session ID`, `Messaging Session Metric Type`,
  `Messaging Session Metric Value`, **`Messaging Channel: Channel Name`** (country source for messaging).
- `report_presence.csv` — `Service Presence Status: Developer Name`, `Status Start/End`, `Is Away`,
  `At Capacity Duration`, `Start DT`, `End DT`.
- `shifts.csv` — wide format, `Agent Name` + one column per date. (Loader: `normalize_wide_shifts`.)
- `survey.csv` — `Survey Taker: Created Date`, `Response`, `Question Type`,
  `Survey Question: Question Title`.
- `chat.csv` — historical Live Chat only.

---

## 2. Confirmed field bindings (the resolved blockers)

### 2.1 Email first-response SLA (Change 1, Change 5)
- **Queue entry = `Date/Time Opened`.** (For email-to-case, case creation ≈ queue entry. Treat as the
  SLA clock start.)
- **First human response = `Completion Date`** for **non-reopened** cases (`Re-opened == 0`).
  Confirmed: it is the milestone-completion/first-response moment, not case closure
  (non-reopened median ≈ 1.4 business-h).
- **Recompute the 4-business-hour SLA ourselves** from `Date/Time Opened` → `Completion Date` using the
  brief's calendar (§2.5). Do **not** trust the export's `Elapsed Time (Hours)` (its business calendar
  is unconfirmed). `Elapsed Time (Hours)` may be kept only as a cross-check.
- Email SLA target: **80% within 4 business hours**. Cohort = queue-entry. Responded within target =
  success; after = fail; unanswered & overdue = fail; unanswered & not-yet-due = **pending, excluded from
  the mature denominator**; latest cohort provisional until all answered-or-overdue.

### 2.2 Reopen (Change 3.3, 5.2, 6) — PARTIALLY available, handle carefully
- **`Re-opened` (0/1) is the authoritative reopen flag.** Use it. **DELETE the old
  `Reopened = Touches > 1` proxy** at `department_dashboard.py` ~line 1785 — the brief explicitly bans
  inferring reopening from routing touches.
- **`Reopened Time`** is stamped by a Salesforce Flow **from 2026-11-19 onward only.**
  - It is **blank for all historical reopened cases** and historical reopen times are **unrecoverable**
    (Status field-history tracking was off).
  - So the **12-hour reopened-resolution clock and reopened-ageing bands (Change 5.2)** and the precise
    **7-day window for Operational FCR (Change 3.3)** can only be computed for reopens occurring after
    2026-11-19. They will fill in week by week.
  - Build these metrics now, but **gate them on `Reopened Time` being present**; where absent, show
    "awaiting data (reopens from 2026-11-19)" rather than a wrong number. Never fabricate from the flag.

### 2.3 Country (Change 6, §10.1 of brief) — RESOLVED
- `email.csv` `Country` is 100% populated. 20 distinct values. Build a controlled mapping; preserve the
  raw value for audit; blank/unmapped → `Unknown`.
- **Mapping table (from the actual 20 values):**
  - `DR Congo` → `Democratic Republic of the Congo`
  - `R Congo` → `Republic of the Congo`
  - `Nigeria - NairaBet` → `Nigeria` (retain `Nigeria - NairaBet` as market label in audit detail)
  - All 17 others are already clean country names, map to themselves:
    `Zimbabwe, Malawi, Liberia, Namibia, Sierra Leone, Cameroon, Uganda, Guinea, Togo, Tanzania, Angola,
    Gabon, Mali, Senegal, Ghana, Central African Republic, Chad`.
- **Messaging country** = `report_messaging_metrics.Messaging Channel: Channel Name`
  (existing helper `_country_from_devname`). Email and Messaging country are separate sources.

### 2.4 Messaging (Change 1, 4) — RESOLVED
- **Handling rate** = `Sessions closed by human / (human-handled + customer-abandoned)` from
  `report_messaging_agent_work.Status`: `Closed` (handled) vs `Abandoned`. Exclude `Canceled`
  (technical cancellation), `Unavailable`, bot-only, test, duplicate, unclassifiable. Target **80%**.
  Display every excluded category + count.
- **First-response (20s clock)** = metric type **`Service Rep Accept To First Response Time`** in
  `report_messaging_metrics` (accept → first response). Target **80% within 20s**. Accepted with no human
  response = fail.
- **Complete-service rule** (Messaging contribution to overall): a session succeeds **only if** handled
  by a human **AND** first-responded within 20s of acceptance. Abandoned = fail.
- **Routing diagnostic** (separate from abandonment): `Assign Date`→`Accept Date`, and `Speed To Answer`.
- **Message counts** (Change 4.2): metric types `Agent Message Count`, `End User Message Count`.

### 2.5 Business calendar (brief §5.2)
- 07:00–22:00 **Central Africa Time (UTC+2)**, **7 days/week**, public holidays are normal operating
  days. Time 22:00–07:00 pauses the clock. Automated acknowledgements do not stop the clock.
- Reporting week Mon–Sun. Refresh weekly Mon, complete through Sun 23:59 CAT.
- Headline comparisons: previous equivalent period, same length. Rate changes shown in **percentage
  points**. Current incomplete periods labelled **provisional**. Overall reporting uses the latest common
  complete date across required sources.

### 2.6 Presence states → adherence (Change 2)
Actual `Service Presence Status: Developer Name` values and their role:
- **Productive (green) readiness:** `Available_All`, `Available_Email_and_Web`, `Available_Chat`
  (= brief "Enhanced Chat"/"Live Agent"), `Available_Case` (= brief "Social Cases").
  > The brief lists 5 green states; the data exposes these 4 `Available_*` dev-names. Map accordingly.
  > Verified active Email/Messaging handling also counts as productive and overrides a conflicting
  > presence status (log the mismatch as a data-quality exception). Merge overlapping productive intervals.
- **Approved exclusions:** `Busy_Meeting`, `Busy_Other_Tasks`.
- **Lunch/Break combined cap 60 min/shift:** `Busy_Lunch`, `Busy_Break`, `On_Break`. Beyond 60 min =
  non-adherent. Unused allowance not auto-deducted.
- `Busy` (bare) / offline during expected productive time = non-adherent.

### 2.7 Survey (Change 3)
`Survey Question: Question Title` values drive classification:
- **CSAT:** "How satisfied were you with the Customer Service you received today?"
  → % rating 4 or 5 of valid respondents. Target 80%. Min sample 30. (Response value holds the rating.)
- **NPS:** "How likely are you to recommend Premier Bet…" **and** "…Mercury Bet…" (two brands — include
  both). %Promoters − %Detractors. Supporting only, no pass/fail. Min 30.
- **Survey FCR:** "Was the issue reported in the chat resolved by the agent?" (chat) and
  "Was the issue reported in the email resolved by the agent?" (email). % confirming resolution. Target
  80%. Min 30. **Channel split is built into the question** (chat vs email).
- Survey **response rate is unavailable** (surveys-issued volume not collected) — state explicitly.
- CSAT has **no channel attribution** in the data → department-total CSAT only (don't invent a channel split).

---

## 3. Per-change build notes (bind brief intent to the data above)

- **Change 1 — Overall Service Attainment.** `(successful eligible Email + successful eligible Messaging)
  / (measurable mature eligible Email + eligible Messaging)`. Target **80%**; overall result alone
  determines pass/fail; Email & Messaging shown but don't veto. Coverage ≥ **95%** to be "reliable";
  below → grey + low-coverage label. Statuses: Green ≥80, Amber 70–79.9, Red <70, Grey <95% coverage or
  immature. Historical Live Chat only in a historical section (pre-migration periods); never in current
  scorecards.
- **Change 2 — Schedule Adherence.** `productive scheduled minutes / expected productive minutes`, target
  **95%**, time-weighted, restricted to scheduled-shift intervals, capped 100%, coverage ≥95%. Attendance
  rules per brief §6.3 (absent = 0%; leave/sick excluded & reported separately; blank/day-off excluded;
  late/early incident only when variance > 5 min, record full variance). Statuses: Green ≥95, Amber
  90–94.9, Red <90, Grey <95% coverage.
- **Change 3 — CX.** CSAT / survey-FCR / operational-FCR / NPS kept **separate**. Operational FCR
  (`resolved not formally reopened within 7 days / resolved with complete 7-day window`) — build using
  the `Re-opened` flag + `Reopened Time`; **gate on reopen-timing availability** (§2.2). Statuses as brief
  §7.5. Show channel survey results only where that channel independently reaches 30.
- **Change 4 — Efficiency/Workload diagnostics.** No pass/fail. Handling time (per case & per touch:
  mean/median/p90/touches/%multi), messages, transfers, utilisation (concurrency = 2 Messaging sessions/
  agent; 1 session = 50%; during `Available_All` any active Messaging removes Email capacity), routing,
  volumes. Diagnostic framing only — do **not** label lower AHT / more messages / transfers good or bad.
- **Change 5 — Email Backlog & Ageing.** New cases: 4h target, bands <4/4–8/8–24/24–48/>48 business-h.
  Reopened cases: 12h clock from first-reopen, bands <12/12–24/24–48/>48 — **gated on `Reopened Time`**.
  Safeguard: ≥80% within target AND zero cases >48h; statuses per brief §9.3. Reconciliation opening + new
  + reopened − resolved = closing.
- **Change 6 — Department breakdowns.** Dimensions: channel; Email Queue (via `report_items.Queue: Name`,
  joined on case number); Email `Case Origin` (`Contact Us` = Contact Form vs `Email` = Direct Email);
  country/market (§2.3); day-of-week & hour-of-day; new vs reopened Email (via `Re-opened` flag);
  transferred vs non-transferred Messaging. Safeguards: ≥30 eligible contacts for a coloured %; coverage
  ≥95%. No agent dimension. Priority excluded (legacy). Adaptive time granularity: daily ≤31d, weekly
  32–180d, monthly >180d; group SLA by arrival, completion separately by response/resolution time.
- **Change 7 — Governance/storage/Git cleanup.** No data blockers. Remove production CSVs from the repo
  + history, `.gitignore` raw exports, load validated weekly snapshots from restricted Google Drive
  (`Current` + `Archive/YYYY-MM-DD`), snapshot manifest, synthetic fixtures only. Drive auth mechanism is
  still an open decision (§5). **Treat the live production CSVs as sensitive — do not commit new copies.**

---

## 4. Known limitations (document these in-app, don't hide them)

1. **Historical reopen timing is empty and unrecoverable.** Reopened-ageing bands and the 12-hour clock
   are blank for periods before 2026-11-19 and fill in weekly thereafter.
2. **Reopened cases lack an initial first-response time.** For `Re-opened == 1`, `Completion Date` is the
   *final* response (median ≈ 5.9 business-h vs 1.4 for non-reopened), not the initial one. So Change 1's
   Email first-response SLA is approximate for the ~11% reopened cases. Chosen handling: **measure on
   best-available `Completion Date` and flag reopened cases** (do NOT silently exclude them — brief §3).
   A future frozen `First Response Date` export field would make this exact (optional).
3. **Survey response rate unavailable** (issued-volume not collected).
4. **CSAT channel split unavailable** (no channel on CSAT rows) — department total only.

---

## 5. Still-open decisions (from brief §14 — not yet resolved; flag, use stated default, don't block)

2. Whether reopened 12h-attainment feeds the overall contact-weighted score, or stays a separate Email
   measure. *Default:* separate Email measure; overall Email = initial 4h first response only.
5. Exact Enhanced Messaging migration date for historical Live Chat routing.
7. Deployment Google Drive auth/secrets mechanism (Change 7).
8. Messaging sessions routed but never accepted with no explicit abandon event: failure / exclusion /
   unmeasurable. *Default:* treat as unmeasurable (exclude from attainment, show in routing diagnostics).
9. Minimum mature-contact volume for headline Email/Messaging/Overall beyond the 95% coverage gate.
10. Survey cohort timestamp (response date / interaction date / other). *Default:* `Survey Taker: Created
    Date`.
11. Operational-FCR cohort timestamp (resolution date / 7-day-window-complete date). *Default:* resolution
    date; pending until window complete.
12. Backlog snapshot semantics (day-end timing; reopened inflow = events or distinct cases; reconciliation
    of opening-backlog cases reopened in-period).
13. Location/runtime for weekly Drive validation + atomic snapshot activation (in-app vs separate ingest).

---

## 6. Acceptance criteria (from brief §15 — the build is done when all hold)

Headline %s reconcile to visible num/den; overall = (success Email + success Messaging) / combined mature
eligible; no current-period Live Chat after migration; missing/zero Messaging responses can't improve SLA;
overdue-unanswered Email = fail, not-yet-due = pending; adherence ignores out-of-shift time and applies the
60-min Lunch/Break cap; **no agent names anywhere**; survey measures obey 30-response thresholds; breakdowns
obey 30-contact + 95%-coverage safeguards; backlog reconciles opening + inflow − outflow = closing; new vs
reopened use separate 4h / 12h clocks; weekly source dates visible and metrics never cross mismatched
cutoffs; a bad source blocks only dependent KPIs; production data absent from GitHub incl. history;
automated tests cover every formula, eligibility edge case, maturity rule and coverage gate.
