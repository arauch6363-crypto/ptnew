# PT — Paris-Turf Prediction Pipeline

Daily horse racing prediction pipeline for French flat racing (Paris-Turf). Scrapes race data, generates ML-based place predictions, and publishes HTML race cards with live PMU odds trends and AI verdicts.

---

## Workflow Overview

```
PT_getData              Scrape today's races + historical results from paris-turf.com
      ↓
PT_Predictor_live       Generate place predictions via LightGBM model
      ↓
PT_Vorarbeiten          Pre-process data, compute ratings, generate AI verdicts,
                        build base HTML files (PMU placeholders)
      ↓
PT_Create_HTMLs_fast    Fetch live PMU odds → update only the odds sections in existing HTMLs
                        (seconds, not minutes)
```

Run the full pipeline automatically via **PT_WORKFLOW.ipynb** (Google Colab, daily trigger via Google Apps Script or manually).

The split between Vorarbeiten and Create_HTMLs_fast means all expensive computation runs once; the odds refresh is near-instant.

---

## File Structure

```
ptnew/
├── PT_WORKFLOW.ipynb              Orchestrator — runs all 4 notebooks via papermill
├── PT_getData.ipynb               Step 1: Web scraping (Selenium)
├── PT_Predictor_live.ipynb        Step 2: ML predictions (LightGBM)
├── PT_Vorarbeiten.ipynb           Step 3: Feature engineering, AI verdicts, base HTML export
├── PT_Create_HTMLs_fast.ipynb     Step 4: PMU odds-only refresh in existing HTMLs
├── pt_html_functions.py           Shared rendering module (HTML, AI verdicts, odds update)
├── going_overrides.ipynb          Manual track condition editor
│
├── scripts/
│   ├── html_fast.py               Railway runner — same job as Create_HTMLs_fast, no Colab
│   └── colab_scheduler.gs         Google Apps Script — daily Colab trigger at 10:00 Berlin
│
├── Dockerfile                     Railway build (python:3.11-slim)
├── railway.toml                   Railway cron: every 10 min from 11:00 Berlin
├── requirements_railway.txt       Python dependencies for Railway
└── nixpacks.toml                  (legacy, superseded by Dockerfile)
```

### Google Drive Layout (`MyDrive/PT/`)

```
PT/
├── races.parquet                  Historical race metadata
├── runners.parquet                Historical runner data
├── odds.parquet                   Historical odds
├── webTips.parquet                Historical expert tips
├── dividends.parquet              Historical dividends
│
├── races_tdy.parquet              Today's races       ← PT_getData
├── runners_tdy.parquet            Today's runners     ← PT_getData
├── odds_tdy.parquet               Today's odds        ← PT_getData
├── webTips_tdy.parquet            Today's tips        ← PT_getData
│
├── today_tips.parquet             Model predictions   ← PT_Predictor_live
│
├── runners_processed.parquet      Processed history   ← PT_Vorarbeiten
├── df_with_ratings.parquet        Elo ratings cache   ← PT_Vorarbeiten
├── ratings_state.json             Latest ratings      ← PT_Vorarbeiten
├── ratings_last_date.txt          Last processed date ← PT_Vorarbeiten
├── notepad_flags_YYYY-MM-DD.pkl   Claude notepad flags← PT_Vorarbeiten
├── precomputed_tdy_YYYY-MM-DD.pkl Today's data + verdicts ← PT_Vorarbeiten
├── pt_html_functions.py           Synced from repo    ← PT_WORKFLOW (git pull → shutil.copy)
│
├── pt_model.pkl                   Trained LightGBM model
├── pt_scaler.pkl                  StandardScaler
├── pt_features.pkl                Feature list (85 features)
│
├── going_overrides.json           Manual going condition overrides (optional)
├── reload_tracker.csv             Scraping progress tracker
│
├── races/                         HTML race cards
│   └── YYYY-MM-DD__Meeting__Race.html
└── workflow_logs/                 Execution logs
    ├── YYYY-MM-DD_workflow.json
    └── YYYY-MM-DD_PT_*_out.ipynb
```

---

## Setup

### 1. Google Colab Secrets

Add these in Colab → **Secrets** (key icon in left sidebar):

| Secret | Used by | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | PT_Vorarbeiten | Claude API — notepad flags + AI verdicts |
| `GITHUB_TOKEN` | PT_Create_HTMLs_fast | Read live odds from `pmu-tracker` repo |

### 2. Required Colab Packages

Installed automatically by the notebooks on first run:
```
selenium, unidecode, fastparquet, google-colab-selenium
anthropic, papermill
```

### 3. One-time Model Setup

Before running the pipeline for the first time:
```
PT_Predictor_train.ipynb → saves pt_model.pkl, pt_scaler.pkl, pt_features.pkl
```

### 4. Railway Setup (automated odds refresh)

For fully automated odds updates without keeping Colab open:

1. Create a [Railway](https://railway.app) project, connect this repo
2. Set environment variables in Railway:
   - `GITHUB_TOKEN` — same token as Colab secret
   - `GOOGLE_CREDENTIALS` — service account JSON (single line, from Google Cloud IAM)
3. Share `MyDrive/PT/` and `MyDrive/PT/races/` with the service account email
4. Railway builds via Dockerfile and runs `scripts/html_fast.py` on cron schedule

Cron schedule (`railway.toml`): `*/10 9-22 * * *` — every 10 minutes from 09:00 UTC (= 11:00 Berlin CEST). Adjust to `*/10 10-22 * * *` in winter (CET).

### 5. Google Apps Script Trigger (automated Colab run)

To run the full pipeline daily at 10:00 Berlin time without manual intervention:

1. Open [script.google.com](https://script.google.com) → New project
2. Paste contents of `scripts/colab_scheduler.gs`
3. Set `NOTEBOOK_FILE_ID` (from the Colab URL: `colab.research.google.com/drive/<ID>`) and `NOTIFY_EMAIL`
4. Run `setupTrigger()` once → grant permissions
5. Run `testRun()` to verify immediately

The script uses the internal Colab kernel API to start a VM and execute all cells. A confirmation email is sent on success or failure.

---

## Running the Pipeline

### Automated — PT_WORKFLOW.ipynb

1. Open `PT_WORKFLOW.ipynb` in Google Colab
2. **Runtime → Run all** (`Ctrl+F9`)

The orchestrator:
- Fetches API secrets in the parent kernel (bypasses papermill secret restriction)
- Clones / pulls this repo to `/content/ptnew`
- Syncs `pt_html_functions.py` from repo → Drive so sub-notebooks use the latest version
- Runs all 4 notebooks sequentially via `papermill`
- Stops on first failure with error details
- Saves a JSON execution log to `workflow_logs/`

**Skip a step:** set `'skip': True` in Cell 3 of PT_WORKFLOW.ipynb.

### Manual — run notebooks individually

```
PT_getData  →  PT_Predictor_live  →  PT_Vorarbeiten  →  PT_Create_HTMLs_fast
```

PT_Create_HTMLs_fast can be re-run any number of times during the day to refresh PMU odds.

---

## Notebook Details

### PT_getData
- Scrapes `paris-turf.com` via Selenium (Chrome)
- Filters: France only, non-trot, flat (`P`), null breed
- Modes: `'get_both'` (today + backfill) or `'get_todays_races'` (today only)
- Progress tracked in `reload_tracker.csv` (safe to restart after interruption)

### PT_Predictor_live
- Loads pre-trained LightGBM model + scaler + feature list
- Computes 85 features: rolling stats per horse/jockey/trainer, t-scores, last-time-out stats
- Outputs `today_tips.parquet` with ranked predictions per race

### PT_Vorarbeiten
- Builds `runners_processed.parquet` (full feature-engineered history)
- Incrementally updates Elo-style horse ratings (`df_with_ratings.parquet`)
- Computes ARR (Adjusted Racing Rating) per runner
- Calls Claude API to flag horses with noteworthy recent runs → `notepad_flags_{TODAY}.pkl`
- **AI verdicts:** calls `generate_race_verdicts()` once per race → Racing Post style
  3–5 sentence verdict per horse, stored in `all_verdicts` dict
- Saves `precomputed_tdy_{TODAY}.pkl` including `all_verdicts`
- **Exports base HTML files** to `Drive/PT/races/` with `<!--PMU_START/END-->` placeholders
  (PMU odds sections are empty at this point, to be filled by Create_HTMLs_fast)

### PT_Create_HTMLs_fast
- **Only task:** fetch live PMU odds + update odds sections in existing HTML files
- Loads `precomputed_tdy` for horse name matching
- Fetches `odds_{TODAY}.json` from `arauch6363-crypto/pmu-tracker` repo
- Calls `update_all_races_html_odds()` — regex-replaces `<!--PMU_START:X-->...<!--PMU_END:X-->` sections
- No re-rendering, no data processing — typically completes in seconds

### scripts/html_fast.py (Railway)
Same job as PT_Create_HTMLs_fast but runs outside Colab:
- Downloads `precomputed_tdy` and today's HTML files from Drive via Google Drive API
- Fetches fresh PMU odds from GitHub
- Calls `update_all_races_html_odds()` locally
- Re-uploads updated HTML files to Drive

### pt_html_functions.py
Shared module loaded via `%run` in Colab or imported in scripts. Key functions:

| Function | Description |
|---|---|
| `export_all_races_html(...)` | Full HTML render for all races; accepts `horse_verdicts` |
| `update_all_races_html_odds(...)` | Fast regex-based PMU odds update in existing files |
| `build_race_json(...)` | Serialises one race to a structured dict for Claude input |
| `generate_race_verdicts(race_json, api_key)` | Calls Claude Sonnet → `{horse_name: verdict}` |
| `compute_notepad_flags(...)` | Claude API batch call → `{(raceId, horseId): True}` |
| `VERDICT_SYSTEM_PROMPT` | Statistics reference + writing style guide for Claude |

---

## HTML Race Card

Each card contains:
- **Header bar** — meeting, race name, class, prize, distance, going
- **Paristurf Verdict** — curated expert tip from webTips
- **Per horse panel:**
  - Name, draw, age, sex, weight, val, rtr, days since last run
  - Equipment/trainer/jockey change indicators
  - PMU live odds strip (`<!--PMU_START:HORSE-->` placeholder, filled by Create_HTMLs_fast)
  - Career stats badges: R/W/P, A/E, €/R with percentile circles
  - Form context table (last N races)
  - Trainer · Jockey · Sire columns with stats and course/distance preferences
  - **AI Verdict** (blue panel, bottom) — Racing Post style, written by Claude Sonnet

### Key Statistics

| Stat | Formula | Meaning |
|---|---|---|
| `val` | `handicapRatingKg − weightKg + 55` | Handicap edge — higher = better positioned |
| `rtr` | `rating_after_race − weightKg + 55` | Weight-adjusted last-run performance |
| `A/E` | actual places ÷ expected places | >1.0 = outperforms odds; <0.9 = underperforms |
| `€/R` | prize per run (median for sires) | Class indicator |
| `pos_perc` | `(pos − 1) ÷ (field − 1)` | 0 = winner, 1 = last; used for draw bias |

---

## Configuration

### going_overrides.json
Override track conditions manually:
```json
{
  "date": "2026-04-23",
  "by_meeting": { "Chantilly": "Bon" },
  "by_race_name": { "Prix de l'Arc": "Lourd" }
}
```
Applied automatically if the date matches today. Edit via `going_overrides.ipynb`.

### Workflow timeouts (PT_WORKFLOW.ipynb Cell 3)

| Step | Default timeout |
|---|---|
| PT_getData | 1800s (30 min) |
| PT_Predictor_live | 900s (15 min) |
| PT_Vorarbeiten | 900s (15 min) |
| PT_Create_HTMLs_fast | 300s (5 min) |

---

## Troubleshooting

**`TimeoutException: Requesting secret ... timed out`**
Always start from `PT_WORKFLOW.ipynb` — it fetches secrets in the parent kernel and passes them as papermill parameters. Secrets cannot be read from sub-kernels in Colab.

**`pt_html_functions.py` changes not taking effect**
PT_WORKFLOW Cell 2 syncs the file from repo → Drive after each git pull. If running notebooks individually, manually copy `pt_html_functions.py` to `MyDrive/PT/`.

**`colab_request` warnings in papermill output**
Harmless. Papermill doesn't recognise Colab-specific IPython messages but continues normally.

**PT_Vorarbeiten fails on ARR / ratings step**
The incremental cache may be out of sync. Delete `df_with_ratings.parquet`, `ratings_state.json`, `ratings_last_date.txt` to force a full recompute.

**AI verdicts missing / empty**
Check that `ANTHROPIC_API_KEY` is set (Colab secret or env var). Verdicts are generated per-race — a failure for one race is logged but doesn't stop the others. Check the Vorarbeiten output notebook in `workflow_logs/`.

**PMU odds not updating**
`update_all_races_html_odds()` requires base HTML files to exist in `Drive/PT/races/`. Run PT_Vorarbeiten first. Files are matched by the pattern `YYYY-MM-DD__*.html`.

**`precomputed_tdy_{TODAY}.pkl` not found**
PT_Vorarbeiten must complete successfully before PT_Create_HTMLs_fast or `scripts/html_fast.py`. Check `workflow_logs/` for the Vorarbeiten output notebook.

**Railway build fails**
Ensure Railway project root is `/` (not `scripts/`). Build uses the `Dockerfile` (`FROM python:3.11-slim`). Check that `GITHUB_TOKEN` and `GOOGLE_CREDENTIALS` environment variables are set in Railway.
