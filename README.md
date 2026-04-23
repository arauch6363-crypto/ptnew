# PT — Paris-Turf Prediction Pipeline

Daily horse racing prediction pipeline for French flat racing (Paris-Turf). Scrapes race data, generates ML-based place predictions, and publishes HTML race cards with live odds trends.

---

## Workflow Overview

```
PT_getData              Scrape today's races + historical results from paris-turf.com
      ↓
PT_Predictor_live       Generate place predictions via LightGBM model
      ↓
PT_Vorarbeiten          Pre-process all data, compute ratings & notepad flags (Claude API)
      ↓
PT_Create_HTMLs_fast    Fetch live PMU odds → build & export HTML race cards
```

Run the full pipeline automatically via **PT_WORKFLOW.ipynb**.

---

## File Structure

```
ptnew/
├── PT_WORKFLOW.ipynb           Orchestrator — runs all 4 notebooks via papermill
├── PT_getData.ipynb            Step 1: Web scraping (Selenium)
├── PT_Predictor_live.ipynb     Step 2: ML predictions (LightGBM)
├── PT_Vorarbeiten.ipynb        Step 3: Feature engineering & pre-computation
├── PT_Create_HTMLs_fast.ipynb  Step 4: HTML export (PMU odds only)
├── pt_html_functions.py        Shared HTML rendering + Claude API notepad flags
└── going_overrides.ipynb       Manual track condition editor
```

### Google Drive Layout (`MyDrive/PT/`)

```
PT/
├── races.parquet               Historical race metadata
├── runners.parquet             Historical runner data
├── odds.parquet                Historical odds
├── webTips.parquet             Historical expert tips
├── dividends.parquet           Historical dividends
│
├── races_tdy.parquet           Today's races      ← written by PT_getData
├── runners_tdy.parquet         Today's runners    ← written by PT_getData
├── odds_tdy.parquet            Today's odds       ← written by PT_getData
├── webTips_tdy.parquet         Today's tips       ← written by PT_getData
│
├── today_tips.parquet          Model predictions  ← written by PT_Predictor_live
│
├── runners_processed.parquet   Processed history  ← written by PT_Vorarbeiten
├── df_with_ratings.parquet     Elo ratings cache  ← written by PT_Vorarbeiten
├── ratings_state.json          Latest ratings     ← written by PT_Vorarbeiten
├── ratings_last_date.txt       Last processed date← written by PT_Vorarbeiten
├── notepad_flags_YYYY-MM-DD.pkl  Claude API flags ← written by PT_Vorarbeiten
├── precomputed_tdy_YYYY-MM-DD.pkl  Processed today← written by PT_Vorarbeiten
│
├── pt_model.pkl                Trained LightGBM model
├── pt_scaler.pkl               StandardScaler
├── pt_features.pkl             Feature list (85 features)
│
├── going_overrides.json        Manual going condition overrides (optional)
├── reload_tracker.csv          Scraping progress tracker
│
├── races/                      Output HTML race cards
│   └── YYYY-MM-DD__Meeting__Race.html
└── workflow_logs/              Execution logs (written by PT_WORKFLOW)
    ├── YYYY-MM-DD_workflow.json
    └── YYYY-MM-DD_PT_*_out.ipynb
```

---

## Setup

### 1. Google Colab Secrets

Add these in Colab → **Secrets** (key icon in left sidebar):

| Secret name | Used by | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | PT_Vorarbeiten | Claude API for notepad flags |
| `GITHUB_TOKEN` | PT_Create_HTMLs_fast | Read live odds from `pmu-tracker` repo |

### 2. Required Colab Packages

Installed automatically by the notebooks on first run:
```
selenium, unidecode, fastparquet, google-colab-selenium
anthropic, papermill
```

### 3. One-time Model Setup

Before running the pipeline for the first time, train and save the model:
```
PT_Predictor_train.ipynb → saves pt_model.pkl, pt_scaler.pkl, pt_features.pkl
```

---

## Running the Pipeline

### Automated (recommended) — PT_WORKFLOW.ipynb

1. Open `PT_WORKFLOW.ipynb` in Google Colab
2. **Runtime → Run all** (`Ctrl+F9`)

The orchestrator:
- Fetches both API secrets in the parent kernel (bypasses papermill secret restriction)
- Clones / updates this repo to `/content/ptnew`
- Runs all 4 notebooks sequentially via `papermill`
- Stops on first failure with error details
- Saves a JSON execution log to `workflow_logs/`

**Skip a step** by setting `'skip': True` in Cell 3 of PT_WORKFLOW.ipynb.

### Manual — run notebooks individually

Open each notebook in Colab and run all cells in order:
```
PT_getData  →  PT_Predictor_live  →  PT_Vorarbeiten  →  PT_Create_HTMLs_fast
```

PT_Create_HTMLs_fast can be re-run any number of times during the day to refresh odds.

---

## Notebook Details

### PT_getData
- Scrapes `paris-turf.com` via Selenium (Chrome)
- Filters: France only, non-trot, flat (`P`), null breed
- Runs in two modes controlled by `mode` variable:
  - `'get_both'` — fetch today's races + backfill missing historical dates
  - `'get_todays_races'` — today only
- Progress tracked in `reload_tracker.csv` (safe to restart)

### PT_Predictor_live
- Loads pre-trained LightGBM model + scaler + feature list
- Computes 85 features: rolling stats per horse/jockey/trainer, t-scores, last-time-out stats
- Outputs `today_tips.parquet` with ranked predictions per race

### PT_Vorarbeiten
- Builds `runners_processed.parquet` (full feature-engineered history)
- Incrementally updates Elo-style horse ratings (`df_with_ratings.parquet`)
- Computes ARR (Adjusted Racing Rating) per runner
- Calls Claude API in batches to flag horses with noteworthy recent runs → `notepad_flags_{TODAY}.pkl`
- Saves `precomputed_tdy_{TODAY}.pkl` — processed today's data ready for HTML export

### PT_Create_HTMLs_fast
- **Only task:** fetch live PMU odds from GitHub + render HTML race cards
- Loads all pre-computed data from PT_Vorarbeiten
- Fetches `odds_{TODAY}.json` from `arauch6363-crypto/pmu-tracker` repo
- Filters to races not yet started (Berlin timezone)
- Exports one HTML file per race to `MyDrive/PT/races/`

### pt_html_functions.py
Shared module loaded via `%run`. Key functions:
- `export_all_races_html(...)` — renders all races to HTML files
- `compute_notepad_flags(df_today, runners_hist)` — Claude API batch call, returns `{(raceId, horseId): True}` for flagged horses
- `stats(df, grouper, ...)` — aggregated performance stats
- `timeseries(df, grouper, ...)` — time-series stats for charts

---

## Configuration

### going_overrides.json
Manually override track conditions for today. Format:
```json
{
  "date": "2026-04-23",
  "by_meeting": {
    "Chantilly": "Bon"
  },
  "by_race_name": {
    "Prix de l'Arc": "Lourd"
  }
}
```
Applied automatically if the date matches today. Edit via `going_overrides.ipynb`.

### Workflow timeouts (PT_WORKFLOW.ipynb, Cell 3)

| Step | Default timeout |
|---|---|
| PT_getData | 1800s (30 min) |
| PT_Predictor_live | 900s (15 min) |
| PT_Vorarbeiten | 900s (15 min) |
| PT_Create_HTMLs_fast | 300s (5 min) |

---

## Troubleshooting

**`TimeoutException: Requesting secret ... timed out`**
Secrets can only be fetched from the Colab UI. Always start the pipeline from `PT_WORKFLOW.ipynb` — it fetches secrets in the parent kernel and passes them to sub-notebooks as papermill parameters.

**`colab_request` warnings in papermill output**
Harmless. Papermill doesn't recognise Colab-specific IPython messages but continues execution. Drive mounts are OS-level FUSE mounts and remain accessible across kernels.

**PT_Vorarbeiten fails on ARR / ratings step**
The incremental cache (`df_with_ratings.parquet`, `ratings_state.json`, `ratings_last_date.txt`) may be out of sync. Delete them to trigger a full recompute — this takes longer but is safe.

**HTML files not updating**
PT_Create_HTMLs_fast filters to races with start time > current Berlin time. Past races are intentionally excluded. Re-run before race start time.

**`precomputed_tdy_{TODAY}.pkl` not found**
PT_Vorarbeiten must complete successfully before PT_Create_HTMLs_fast. Check `workflow_logs/` for the Vorarbeiten output notebook to diagnose the failure.
