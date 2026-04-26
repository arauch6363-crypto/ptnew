"""
pt_html_functions.py
────────────────────
Shared HTML-rendering module for the PT racing analysis pipeline.

Usage
-----
    from pt_html_functions import (
        TF,
        COLUMNS, COL_LABELS, CHART_METRICS, PERIODS, LINE_COLOURS,
        stats, timeseries,
        compute_notepad_flags,
        export_all_races_html,
        _render_runners_html,   # internal, but importable if needed
        generate_race_verdicts,
        generate_race_verdict,
        update_verdicts_in_html,
        update_race_verdicts_in_html,
    )

This file is the single source of truth for all rendering logic.
Both PT_Vorarbeiten.ipynb and PT_Create_HTMLs_fast.ipynb import from here
so they never go out of sync.

Drop this file into BASE (/content/drive/MyDrive/PT/) before running
either notebook.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import math
import bisect as _bisect
import datetime as _dt
import unicodedata
import warnings
from pathlib import Path


def _load_prompt(filename: str) -> str:
    """Load a system prompt — local prompts/ dir first, GitHub raw URL as fallback."""
    local = Path(__file__).parent / 'prompts' / filename
    if local.exists():
        return local.read_text(encoding='utf-8')
    # Fallback: fetch from GitHub (Colab / any env where prompts/ isn't next to this file)
    import urllib.request as _urlreq
    _token = os.environ.get('GITHUB_TOKEN', '')
    _url   = f'https://raw.githubusercontent.com/arauch6363-crypto/ptnew/main/prompts/{filename}'
    _req   = _urlreq.Request(_url, headers={'Authorization': f'token {_token}'} if _token else {})
    try:
        with _urlreq.urlopen(_req, timeout=15) as _r:
            _text = _r.read().decode('utf-8')
        print(f'[pt_html_functions] loaded {filename} from GitHub')
        return _text
    except Exception as _e:
        raise FileNotFoundError(
            f"Cannot load prompt '{filename}': not found at {local} and GitHub fetch failed ({_e}). "
            f"Either place the prompts/ folder next to pt_html_functions.py or set GITHUB_TOKEN."
        ) from _e

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Constants, config helpers, stats functions
# (originally "## 9. Prepare HTML", cell 31)
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = ['runs', 'runners', 'wins', 'win_perc', 'places', 'place_perc',
           'prizemoney', 'ae_win', 'ae_place']

COL_LABELS = {
    'runs'       : 'Runs',
    'runners'    : 'Runners',
    'wins'       : 'Wins',
    'win_perc'   : 'Win %',
    'places'     : 'Places',
    'place_perc' : 'Place %',
    'prizemoney' : 'Prize/Run',
    'ae_win'     : 'A/E Win',
    'ae_place'   : 'A/E Place',
}

CHART_METRICS = [
    ('Runs',      'runs',        'count'),
    ('Wins',      'wins',        'count'),
    ('Win %',     'win_perc',    'pct'),
    ('Place %',   'place_perc',  'pct'),
    ('A/E Win',   'ae_win',      'decimal'),
    ('A/E Place', 'ae_place',    'decimal'),
    ('Prize/Run', 'prizemoney',  'euro'),
]

PERIODS = [
    ('Day',     'D',  '%d %b %y'),
    ('Week',    'W',  'W%W %y'),
    ('Month',   'M',  '%b %y'),
    ('Quarter', 'Q',  'Q%q %y'),
    ('Year',    'Y',  '%Y'),
]

LINE_COLOURS = [
    '#00a651', '#e6194b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#fabed4', '#469990', '#dcbeff', '#1a2b4a',
]

_date_cache = {}

# ── System prompts — loaded from prompts/ directory ──────────────────────────
COMBINED_VERDICT_SYSTEM_PROMPT = _load_prompt('combined_verdict_system_prompt.txt')


def _ensure_date_col(df):
    key = id(df)
    if key not in _date_cache or '_dt' not in df.columns:
        df['_dt'] = pd.to_datetime(df['date'])
        _date_cache[key] = True
    return df

def _filter(df, start, end, odds_min, odds_max, extra_filters=None):
    df = _ensure_date_col(df)
    mask = (
        df['liveOdd'].notnull() &
        df['odds_sum'].between(odds_min, odds_max) &
        (df['_dt'] >= start) &
        (df['_dt'] < end)
    )
    if extra_filters:
        for col, vals in extra_filters.items():
            if vals and col in df.columns:
                mask = mask & df[col].isin(vals)
    return df.loc[mask].copy()

def _filter_to_entities(df, grouper, entity_set):
    if not entity_set or grouper not in df.columns:
        return df
    return df[df[grouper].isin(entity_set)]

def stats(df, grouper='trainerName', start='2025-01-01', end='2026-01-01',
          odds_min=1.1, odds_max=1.4, min_runs=20, extra_filters=None, entity_set=None):
    filtered = _filter(df, start, end, odds_min, odds_max, extra_filters)
    filtered = _filter_to_entities(filtered, grouper, entity_set)
    agg = filtered.groupby(grouper).agg(
        runs=('win', 'count'),
        wins=('win', 'sum'),
        places=('place', 'sum'),
        odds_win_sum=('odds', 'sum'),
        odds_place_sum=('odds_place', 'sum'),
        prizemoney=('prizemoney', 'sum')
    ).astype({'runs': int, 'wins': int, 'places': int})
    agg['runners']    = (filtered[filtered['win'] == 1]
                         .groupby(grouper)['horseId'].nunique()
                         .reindex(agg.index, fill_value=0).astype(int))
    agg['win_perc']   = np.round(agg['wins']   / agg['runs'], 3) * 100
    agg['place_perc'] = np.round(agg['places'] / agg['runs'], 3) * 100
    agg['ae_win']     = np.round(agg['wins']   / agg['odds_win_sum'],   3)
    agg['ae_place']   = np.round(agg['places'] / agg['odds_place_sum'], 3)
    agg['prizemoney'] = np.round(agg['prizemoney'] / agg['runs'], 1)
    return (
        agg[agg['runs'] >= min_runs]
        .drop(columns=['odds_win_sum', 'odds_place_sum'])
        [COLUMNS]
        .sort_values('runners', ascending=False)
    )

def _ts_for_freq(filtered, grouper, freq, start, end):
    df2 = filtered.copy()
    df2['_period'] = df2['_dt'].dt.to_period(freq)
    all_periods    = pd.period_range(start=start, end=end, freq=freq)
    agg = (df2.groupby([grouper, '_period'], observed=True)
              .agg(runs=('win','count'), wins=('win','sum'), places=('place','sum'),
                   odds_win_sum=('odds','sum'), odds_place_sum=('odds_place','sum'),
                   prizemoney=('prizemoney','sum'))
              .reindex(pd.MultiIndex.from_product(
                   [df2[grouper].unique(), all_periods],
                   names=[grouper, '_period']), fill_value=0))
    wins_s   = agg['wins'].astype(float)
    runs_s   = agg['runs'].replace(0, np.nan)
    places_s = agg['places'].astype(float)
    agg['win_perc']   = (wins_s / runs_s * 100).round(1).fillna(0)
    agg['place_perc'] = (places_s / runs_s * 100).round(1).fillna(0)
    agg['ae_win']     = (wins_s / agg['odds_win_sum'].replace(0, np.nan)).round(3).fillna(0)
    agg['ae_place']   = (places_s / agg['odds_place_sum'].replace(0, np.nan)).round(3).fillna(0)
    agg['prizemoney'] = (agg['prizemoney'] / runs_s).round(1).fillna(0)
    if freq == 'Q':
        labels = [f'Q{p.quarter} {p.year}' for p in all_periods]
    else:
        fmt_map = {k: v for _, k, v in PERIODS}
        labels  = [p.start_time.strftime(fmt_map[freq]) for p in all_periods]
    agg['prize_sum'] = agg['prizemoney'] * runs_s.fillna(0)
    METRICS = ['runs','wins','places','win_perc','place_perc','ae_win','ae_place','prizemoney']
    RAW     = ['odds_win_sum', 'odds_place_sum', 'prize_sum']
    entities = {}
    for name, grp in agg.groupby(level=0):
        grp = grp.droplevel(0)
        entities[str(name)] = {'labels': labels}
        for m in METRICS + RAW:
            entities[str(name)][m] = [round(float(x), 3) for x in grp[m]]
    return entities

def timeseries(df, grouper='trainerName', start='2025-01-01', end='2026-01-01',
               odds_min=1.1, odds_max=1.4, extra_filters=None, entity_set=None):
    filtered = _filter(df, start, end, odds_min, odds_max, extra_filters)
    filtered = _filter_to_entities(filtered, grouper, entity_set)
    return {freq: _ts_for_freq(filtered, grouper, freq, start, end)
            for _, freq, _ in PERIODS}

def _rdylgn(v, vmin=0.5, vmax=1.5):
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    if t < 0.5:
        r, g = 220, int(220 * t * 2)
    else:
        r, g = int(220 * (1 - t) * 2), 220
    return f'rgb({r},{g},60)'

def _blues(v, vmax):
    if vmax == 0:
        return 'rgb(247,251,255)'
    t = max(0.0, min(1.0, v / vmax))
    val = int(247 - t * 160)
    return f'rgb({val},{int(val * 0.93)},255)'

def _bar_html(value, max_val, color, fmt):
    pct = max(0.0, min(100.0, value / max_val * 100)) if max_val else 0
    return (
        f'<div style="position:relative;min-width:80px">'
        f'<div style="position:absolute;top:0;left:0;height:100%;'
        f'width:{pct:.1f}%;background:{color};border-radius:2px;opacity:0.7"></div>'
        f'<span style="position:relative">{fmt}</span>'
        f'</div>'
    )

def _build_html(result, all_ts, grouper, start, end, odds_min, odds_max,
                top_n=20, standalone=False, caption_prefix=None):
    _base = f'{grouper} | Odds {odds_min}–{odds_max} | {start} → {end}'
    caption = f'{caption_prefix}  ·  {_base}' if caption_prefix else _base

    max_runs   = result['runs'].max()   if len(result) else 1
    max_wins   = result['wins'].max()   if len(result) else 1
    max_places = result['places'].max() if len(result) else 1
    max_prize  = result['prizemoney'].max() if len(result) else 1

    rows_data = []
    for name, row in result.iterrows():
        rows_data.append({
            '__name__'  : str(name),
            'runs'      : int(row['runs']),
            'runners'   : int(row['runners']),
            'wins'      : int(row['wins']),
            'win_perc'  : float(row['win_perc']),
            'places'    : int(row['places']),
            'place_perc': float(row['place_perc']),
            'prizemoney': float(row['prizemoney']),
            'ae_win'    : float(row['ae_win']),
            'ae_place'  : float(row['ae_place']),
        })

    def cell_html(col, row):
        v = row[col]
        if col in ('ae_win', 'ae_place'):
            bg = _rdylgn(v)
            return f'<td style="background:{bg};text-align:center">{v:.2f}</td>'
        if col in ('runs', 'wins', 'places'):
            mx = {'runs': max_runs, 'wins': max_wins, 'places': max_places}[col]
            bg = _blues(v, mx)
            return f'<td style="background:{bg};text-align:center">{int(v):,}</td>'
        if col == 'runners':
            return f'<td style="text-align:center">{int(v):,}</td>'
        if col == 'win_perc':
            return f'<td>{_bar_html(v, 40, TF["bar_win"],   f"{v:.1f}%")}</td>'
        if col == 'place_perc':
            return f'<td>{_bar_html(v, 40, TF["bar_win"],   f"{v:.1f}%")}</td>'
        if col == 'prizemoney':
            return f'<td>{_bar_html(v, max_prize, TF["bar_prize"], f"€{v:,.0f}")}</td>'
        return f'<td style="text-align:center">{v}</td>'

    uid = f'tf_{abs(hash(caption)) % 10**8}'

    rows_html_map = []
    for i, (name, row) in enumerate(result.iterrows()):
        bg        = TF['row_alt'] if i % 2 == 0 else 'white'
        cells     = ''.join(cell_html(c, row) for c in COLUMNS)
        safe_name = str(name).replace("'", "\\'")
        rows_html_map.append(
            f'<tr data-idx="{i}" data-bg="{bg}" data-name="{str(name)}" '
            f'style="background:{bg}" '
            f'onmouseover="this.style.background=\'{TF["row_hover"]}\'" '
            f'onmouseout="this.style.background=this.dataset.bg">'
            f'<th onclick="tfToggleEntity(\'{safe_name}\', {i})" '
            f'style="background:{TF["row_head_bg"]};color:{TF["row_head_fg"]};'
            f'text-align:left;padding:6px 14px;border-right:2px solid {TF["border"]};'
            f'font-size:13px;font-weight:bold;cursor:pointer;white-space:nowrap;'
            f'user-select:none" title="Click to toggle on chart">'
            f'<span class="tf-swatch" data-row="{i}" '
            f'style="display:inline-block;width:11px;height:11px;border-radius:50%;'
            f'border:2px solid #bbb;margin-right:7px;vertical-align:middle;'
            f'background:transparent;transition:all .15s ease;flex-shrink:0"></span>'
            f'{name}</th>'
            f'{cells}</tr>'
        )

    thead_cells = ''.join(
        f'<th onclick="tfSort(\'{col}\')" style="cursor:pointer" title="Sort by {COL_LABELS[col]}">'
        f'{COL_LABELS[col]} <span id="arr_{col}"></span></th>'
        for col in COLUMNS
    )

    metric_buttons = ''.join(
        f'<button onclick="tfSetMetric(\'{key}\')" id="mbtn_{key}" '
        f'style="margin:2px 3px;padding:4px 10px;font-size:11px;border-radius:3px;'
        f'border:1px solid #ccc;background:#f8f9fa;cursor:pointer;font-family:inherit">'
        f'{label}</button>'
        for label, key, _ in CHART_METRICS
    )

    period_buttons = ''.join(
        f'<button onclick="tfSetPeriod(\'{freq}\')" id="pbtn_{freq}" '
        f'style="margin:2px 3px;padding:3px 9px;font-size:11px;border-radius:3px;'
        f'border:1px solid #ccc;background:#f8f9fa;cursor:pointer;font-family:inherit">'
        f'{label}</button>'
        for label, freq, _ in PERIODS
    )

    default_freq   = 'M'
    active_names   = {r['__name__'] for r in rows_data}
    all_ts_trimmed = {
        freq: {k: v for k, v in ents.items() if k in active_names}
        for freq, ents in all_ts.items()
    }

    rows_json      = json.dumps(rows_data)
    rows_html_json = json.dumps(rows_html_map)
    all_ts_json    = json.dumps(all_ts_trimmed)
    metrics_json   = json.dumps([{'label': l, 'key': k, 'fmt': f} for l, k, f in CHART_METRICS])
    cols_json      = json.dumps(COLUMNS)
    colours_json   = json.dumps(LINE_COLOURS)

    pre  = '<!DOCTYPE html><html><head><meta charset="utf-8"><title>TF Stats</title></head><body style="margin:16px;background:#f0f2f5">' if standalone else ''
    post = '</body></html>' if standalone else ''

    html = f"""{pre}
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
#{uid}_wrap {{
  display: flex; gap: 20px; align-items: stretch;
  font-family: "Helvetica Neue", Arial, sans-serif;
}}
#{uid} {{ min-height: 700px; display: flex; flex-direction: column; }}
#{uid} table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
#{uid} thead th {{
    background: {TF['header_bg']}; color: {TF['header_fg']};
    font-size: 12px; font-weight: bold; text-align: center;
    padding: 10px 12px; border-bottom: 3px solid {TF['subheader_bg']};
    text-transform: uppercase; letter-spacing: .06em;
    user-select: none; white-space: nowrap;
}}
#{uid} thead th:hover {{ background: {TF['subheader_bg']}; }}
#{uid} td, #{uid} th {{ border: 1px solid {TF['border']}; padding: 6px 12px; }}
#{uid} caption {{
    caption-side: top; font-size: 14px; font-weight: bold;
    color: {TF['caption_fg']}; padding: 8px 0 6px 0;
    text-align: left; border-bottom: 2px solid {TF['green']}; margin-bottom: 4px;
}}
.tf-panel {{
  background: white; border-radius: 6px;
  box-shadow: 0 2px 10px rgba(26,43,74,.12);
  padding: 16px; border-top: 4px solid {TF['green']};
}}
.tf-btn-active {{
  background: {TF['navy']} !important; color: white !important;
  border-color: {TF['navy']} !important; font-weight: bold;
}}
#{uid}_chart_panel {{
  display: flex; flex-direction: column;
  min-width: 380px; min-height: 700px; flex: 1;
}}
#{uid}_canvas_wrap {{ flex: 1; position: relative; min-height: 560px; }}
</style>

<div id="{uid}_wrap">
  <div id="{uid}" class="tf-panel" style="overflow-x:auto;flex-shrink:0">
    <table>
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th style="cursor:default;background:{TF['header_bg']}"></th>
          {thead_cells}
        </tr>
      </thead>
      <tbody id="{uid}_body">
        {''.join(rows_html_map[:top_n])}
      </tbody>
    </table>
    <div style="margin-top:6px;font-size:11px;color:#888;font-style:italic">
      Click a name to add / remove from chart &nbsp;·&nbsp;
      <a href="#" onclick="tfClearAll();return false"
         style="color:{TF['green']};text-decoration:none;font-style:normal">Clear all</a>
    </div>
  </div>

  <div id="{uid}_chart_panel" class="tf-panel">
    <div style="display:flex;align-items:center;justify-content:space-between;
                border-bottom:2px solid {TF['green']};padding-bottom:6px;margin-bottom:10px">
      <span style="font-size:14px;font-weight:bold;color:{TF['navy']}">Trend</span>
      <span id="{uid}_hint" style="font-size:11px;color:#888;font-style:italic">Select rows to compare</span>
    </div>
    <div id="{uid}_metric_btns" style="margin-bottom:6px">{metric_buttons}</div>
    <div style="margin-bottom:12px;border-top:1px solid {TF['border']};padding-top:6px;
                display:flex;flex-wrap:wrap;align-items:center;gap:8px">
      <div id="{uid}_period_btns">
        <span style="font-size:11px;color:#888;margin-right:4px">Period:</span>
        {period_buttons}
      </div>
      <div style="display:flex;align-items:center;gap:6px;margin-left:8px;
                  border-left:1px solid {TF['border']};padding-left:12px">
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;
                       font-size:11px;color:#555;user-select:none">
          <input type="checkbox" id="{uid}_roll_toggle"
                 onchange="tfToggleRolling(this.checked)"
                 style="cursor:pointer;accent-color:{TF['green']}">
          Rolling
        </label>
        <div id="{uid}_roll_wrap" style="display:none;align-items:center;gap:4px">
          <input type="number" id="{uid}_roll_n" value="4" min="2" max="365"
                 oninput="tfSetRollingN(+this.value)"
                 style="width:52px;padding:2px 5px;font-size:11px;border:1px solid #ccc;
                        border-radius:3px;text-align:center">
          <span id="{uid}_roll_unit" style="font-size:11px;color:#888">months</span>
        </div>
      </div>
    </div>
    <div id="{uid}_canvas_wrap">
      <canvas id="{uid}_canvas" style="width:100%;height:100%"></canvas>
      <div id="{uid}_no_data"
           style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                  text-align:center;color:#bbb;font-size:13px;font-style:italic;
                  pointer-events:none">
        Click one or more names to plot their trend
      </div>
    </div>
  </div>
</div>

<script>
(function(){{
  var rows={rows_json}; var rowsHtml={rows_html_json}; var allTs={all_ts_json};
  var metrics={metrics_json}; var cols={cols_json}; var colours={colours_json};
  var uid='{uid}'; var topN={top_n};
  var sortCol='runners',sortAsc=false,curMetric='wins',curPeriod='{default_freq}';
  var rollingOn=false,rollingN=4,chart=null,selected={{}},colourIdx=0;
  var PERIOD_UNITS={{D:'days',W:'weeks',M:'months',Q:'quarters',Y:'years'}};

  function updateRollUnit(){{
    var el=document.getElementById(uid+'_roll_unit');
    if(el) el.textContent=PERIOD_UNITS[curPeriod]||'periods';
  }}
  function rollingSum(arr,win){{
    var out=[];
    for(var i=0;i<arr.length;i++){{
      var lo=Math.max(0,i-win+1),s=0;
      for(var j=lo;j<=i;j++) s+=arr[j];
      out.push(Math.round(s*1000)/1000);
    }}
    return out;
  }}
  var METRIC_ROLL={{
    runs:{{type:'count'}},wins:{{type:'count'}},places:{{type:'count'}},
    win_perc:{{type:'ratio',num:'wins',den:'runs',scale:100}},
    place_perc:{{type:'ratio',num:'places',den:'runs',scale:100}},
    ae_win:{{type:'ratio',num:'wins',den:'odds_win_sum',scale:1}},
    ae_place:{{type:'ratio',num:'places',den:'odds_place_sum',scale:1}},
    prizemoney:{{type:'ratio',num:'prize_sum',den:'runs',scale:1}},
  }};
  function rolledSeries(d,metric,win){{
    var spec=METRIC_ROLL[metric];
    if(!spec||!rollingOn||win<2) return d[metric]||[];
    if(spec.type==='count') return rollingSum(d[metric]||[],win);
    var numR=rollingSum(d[spec.num]||[],win);
    var denR=rollingSum(d[spec.den]||[],win);
    var out=[];
    for(var i=0;i<numR.length;i++){{
      var v=denR[i]>0?(numR[i]/denR[i])*spec.scale:0;
      out.push(Math.round(v*1000)/1000);
    }}
    return out;
  }}
  function fmtVal(v,fmt){{
    if(fmt==='pct') return v.toFixed(1)+'%';
    if(fmt==='decimal') return v.toFixed(2);
    if(fmt==='euro') return '\u20ac'+Number(v.toFixed(0)).toLocaleString();
    return String(v);
  }}
  function drawChart(){{
    var names=Object.keys(selected);
    var noData=document.getElementById(uid+'_no_data');
    var hint=document.getElementById(uid+'_hint');
    if(names.length===0){{
      if(chart){{chart.destroy();chart=null;}}
      noData.style.display='block';
      hint.textContent='Select rows to compare'; return;
    }}
    noData.style.display='none';
    hint.textContent=names.length+' series · '+curPeriod+' · '+curMetric;
    var tsForPeriod=allTs[curPeriod]||{{}};
    var firstData=tsForPeriod[names[0]]||{{}};
    var rawLabels=firstData.labels||[];
    var crop=(rollingOn&&rollingN>=2)?Math.min(rollingN,rawLabels.length):0;
    var labels=rawLabels.slice(crop);
    var metaObj=null;
    for(var i=0;i<metrics.length;i++){{if(metrics[i].key===curMetric){{metaObj=metrics[i];break;}}}}
    var fmt=metaObj?metaObj.fmt:'count';
    var isAE=(curMetric==='ae_win'||curMetric==='ae_place');
    var datasets=names.map(function(name){{
      var d=(tsForPeriod[name]||{{}});
      var col=selected[name];
      var r=parseInt(col.slice(1,3),16),g=parseInt(col.slice(3,5),16),b=parseInt(col.slice(5,7),16);
      var bgCol='rgba('+r+','+g+','+b+',0.07)';
      return{{label:name,data:rolledSeries(d,curMetric,rollingN).slice(crop),
              borderColor:col,backgroundColor:bgCol,pointRadius:0,pointHoverRadius:0,
              borderWidth:names.length===1?2.5:2,tension:0.35,fill:names.length===1}};
    }});
    if(isAE) datasets.push({{label:'Expected (1.0)',
      data:labels.map(function(){{return 1.0;}}),
      borderColor:'rgba(180,40,40,0.45)',borderDash:[5,4],borderWidth:1.5,
      pointRadius:0,pointHoverRadius:0,fill:false}});
    if(chart){{chart.destroy();chart=null;}}
    var wrap=document.getElementById(uid+'_canvas_wrap');
    var oldCanvas=document.getElementById(uid+'_canvas');
    var newCanvas=document.createElement('canvas');
    newCanvas.id=uid+'_canvas';newCanvas.style.cssText='width:100%;height:100%;display:block';
    wrap.replaceChild(newCanvas,oldCanvas);
    chart=new Chart(newCanvas.getContext('2d'),{{
      type:'line',data:{{labels:labels,datasets:datasets}},
      options:{{responsive:true,maintainAspectRatio:false,
        interaction:{{mode:'index',intersect:false}},
        plugins:{{
          legend:{{display:names.length>1||isAE,labels:{{font:{{size:11}},boxWidth:18}}}},
          tooltip:{{callbacks:{{label:function(c){{return ' '+c.dataset.label+': '+fmtVal(c.parsed.y,fmt);}}}}}}
        }},
        scales:{{
          x:{{ticks:{{font:{{size:10}},maxRotation:45,autoSkip:true,maxTicksLimit:24}},
              grid:{{color:'rgba(0,0,0,0.04)'}}}},
          y:{{beginAtZero:!isAE,
              ticks:{{font:{{size:10}},callback:function(v){{return fmtVal(v,fmt);}}}},
              grid:{{color:'rgba(0,0,0,0.06)'}}}}
        }}
      }}
    }});
  }}
  function refreshSwatches(){{
    var body=document.getElementById(uid+'_body');
    Array.from(body.querySelectorAll('tr')).forEach(function(tr){{
      var name=tr.dataset.name;
      var sw=tr.querySelector('.tf-swatch');
      if(!sw) return;
      if(selected[name]){{
        var col=selected[name];
        sw.style.background=col;sw.style.borderColor=col;
        sw.style.boxShadow='0 0 0 2.5px '+col+'55';sw.style.transform='scale(1.15)';
      }}else{{
        sw.style.background='transparent';sw.style.borderColor='#bbb';
        sw.style.boxShadow='none';sw.style.transform='scale(1)';
      }}
    }});
  }}
  window.tfToggleEntity=function(name){{
    if(selected[name]) delete selected[name];
    else selected[name]=colours[colourIdx++%colours.length];
    refreshSwatches();drawChart();
  }};
  window.tfClearAll=function(){{selected={{}};colourIdx=0;refreshSwatches();drawChart();}};
  window.tfSetMetric=function(key){{
    curMetric=key;
    document.getElementById(uid+'_metric_btns').querySelectorAll('button')
      .forEach(function(b){{b.classList.remove('tf-btn-active');}});
    var btn=document.getElementById('mbtn_'+key);
    if(btn) btn.classList.add('tf-btn-active');
    drawChart();
  }};
  window.tfSetPeriod=function(freq){{
    curPeriod=freq;
    document.getElementById(uid+'_period_btns').querySelectorAll('button')
      .forEach(function(b){{b.classList.remove('tf-btn-active');}});
    var btn=document.getElementById('pbtn_'+freq);
    if(btn) btn.classList.add('tf-btn-active');
    updateRollUnit();drawChart();
  }};
  window.tfToggleRolling=function(on){{
    rollingOn=on;
    var wrap=document.getElementById(uid+'_roll_wrap');
    if(wrap) wrap.style.display=on?'flex':'none';
    updateRollUnit();drawChart();
  }};
  window.tfSetRollingN=function(n){{if(n>=2){{rollingN=n;drawChart();}}}};
  window.tfSort=function(col){{
    if(sortCol===col) sortAsc=!sortAsc; else{{sortCol=col;sortAsc=false;}}
    cols.forEach(function(c){{var el=document.getElementById('arr_'+c);if(el) el.textContent='';}});
    var arr=document.getElementById('arr_'+col);
    if(arr) arr.textContent=sortAsc?' \u25b2':' \u25bc';
    var order=rows.map(function(r,i){{return{{v:r[col],i:i}};}})
      .sort(function(a,b){{return sortAsc?(a.v>b.v?1:-1):(a.v<b.v?1:-1);}})
      .slice(0,topN);
    var body=document.getElementById(uid+'_body');
    var frag=document.createDocumentFragment();
    order.forEach(function(item){{
      var tmp=document.createElement('tbody');
      tmp.innerHTML=rowsHtml[item.i];
      frag.appendChild(tmp.firstChild);
    }});
    body.innerHTML='';body.appendChild(frag);refreshSwatches();
  }};
  tfSetMetric('wins');
  tfSetPeriod('{default_freq}');
  document.getElementById('arr_runners').textContent=' \u25bc';
  var firstRow=document.querySelector('#{uid}_body tr');
  if(firstRow&&firstRow.dataset.name) tfToggleEntity(firstRow.dataset.name);
}})();
</script>
{post}"""
    return html

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — TF theme dict, SYSTEM_PROMPT_BATCH, and the three main functions:
#   compute_notepad_flags()
#   export_all_races_html()
#   _render_runners_html()
# (originally "## 9. Prepare HTML", cell 32)
# ─────────────────────────────────────────────────────────────────────────────

TF = dict(
    header_bg    = '#1a2b4a',
    header_fg    = '#ffffff',
    subheader_bg = '#2e4a6e',
    row_alt      = '#f4f7fb',
    row_hover    = '#e6f0fa',
    border       = '#d0d9e8',
    caption_fg   = '#1a2b4a',
    row_head_bg  = '#eef2f8',
    row_head_fg  = '#1a2b4a',
    bar_win      = 'rgba(0, 166, 81, 0.45)',
    bar_prize    = 'rgba(0, 122, 204, 0.30)',
    green        = '#00a651',
    navy         = '#1a2b4a',
)

SYSTEM_PROMPT_BATCH = _load_prompt('notepad_batch_system_prompt.txt')

def compute_notepad_flags(df_today, runners_hist, max_races_per_horse=3):
    """
    Pre-computes notepad flags for the last `max_races_per_horse` races of every
    horse running today.  Returns a dict keyed by (raceId, horseId) -> bool.

    Parameters
    ----------
    df_today      : today's race DataFrame (needs horseId, raceId columns)
    runners_hist  : historical race DataFrame (needs horseId, raceId, date, comment, horseName)
    max_races_per_horse : how many recent races per horse to analyse (default 3)

    Returns
    -------
    dict  {(raceId, horseId): True}   — only flagged entries are stored
    """
    import json
    import anthropic
    import datetime as _dt

    notepad_flags = {}

    # ── guard: required columns ───────────────────────────────────────────────
    required_hist = {'horseId', 'raceId', 'date', 'comment', 'horseName'}
    if runners_hist is None or not required_hist.issubset(runners_hist.columns):
        missing = required_hist - set(runners_hist.columns if runners_hist is not None else [])
        print(f'⚠️  compute_notepad_flags: missing columns in runners_hist: {missing}')
        return notepad_flags

    if 'horseId' not in df_today.columns:
        print('⚠️  compute_notepad_flags: horseId missing from df_today')
        return notepad_flags

    today_str  = _dt.date.today().strftime('%Y-%m-%d')
    horse_ids  = df_today['horseId'].dropna().unique().tolist()

    # ── build per-horse last-N-races lookup ───────────────────────────────────
    hist = runners_hist.copy()
    hist['_npdt'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist[hist['_npdt'] < today_str].dropna(subset=['comment'])
    hist = hist[hist['comment'].astype(str).str.strip() != '']
    hist = hist[hist['horseId'].isin(horse_ids)]
    hist = hist.sort_values('_npdt', ascending=False)

    # collect: { raceId -> { horseId -> {horseName, comment, date} } }
    race_horse_map = {}   # raceId -> list of {horseId, horseName, comment}
    race_date_map  = {}   # raceId -> date string

    for hid, grp in hist.groupby('horseId'):
        recent = grp.drop_duplicates(subset=['raceId']).head(max_races_per_horse)
        for _, row in recent.iterrows():
            rid   = row['raceId']
            hname = str(row.get('horseName', '') or '')
            comm  = str(row['comment']).strip()
            dstr  = row['_npdt'].strftime('%Y-%m-%d') if pd.notna(row['_npdt']) else ''
            if not comm or comm in ('nan', '—', ''):
                continue
            if rid not in race_horse_map:
                race_horse_map[rid] = []
                race_date_map[rid]  = dstr
            race_horse_map[rid].append({
                'horseId':   hid,
                'horseName': hname,
                'comment':   comm,
            })

    if not race_horse_map:
        print('ℹ️  compute_notepad_flags: no comment data found for today\'s horses')
        return notepad_flags

    # ── chunk races into batches to stay within token limits ─────────────────
    BATCH_SIZE = 20   # races per API call — tune if needed
    race_ids   = list(race_horse_map.keys())
    batches    = [race_ids[i:i+BATCH_SIZE] for i in range(0, len(race_ids), BATCH_SIZE)]

    import os as _os
    _api_key = _os.environ.get('ANTHROPIC_API_KEY')
    if not _api_key:
        from google.colab import userdata
        _api_key = userdata.get('ANTHROPIC_API_KEY')
    client = anthropic.Anthropic(api_key=_api_key)

    total_flagged = 0

    print(f'🤖 compute_notepad_flags: {len(race_ids)} races → {len(batches)} API call(s)')

    for batch_idx, batch_race_ids in enumerate(batches):
        # build input JSON:  { raceId: [ {horseName, comment}, ... ] }
        payload = {}
        horse_id_lookup = {}   # (raceId, horseName) -> horseId  (for result mapping)
        for rid in batch_race_ids:
            entries = race_horse_map[rid]
            payload[str(rid)] = [
                {'horseName': e['horseName'], 'comment': e['comment']}
                for e in entries
            ]
            for e in entries:
                try:
                    normalized_hid = str(int(float(e['horseId'])))
                except (ValueError, TypeError):
                    normalized_hid = str(e['horseId'])
                horse_id_lookup[(str(rid), e['horseName'])] = normalized_hid

        user_msg = json.dumps(payload, ensure_ascii=False)

        try:
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=4096,
                system=SYSTEM_PROMPT_BATCH,
                messages=[{'role': 'user', 'content': user_msg}],
            )
            raw = response.content[0].text.strip()
            if response.stop_reason == 'max_tokens':
                print(
                    f'  ⚠️  TOKEN LIMIT: notepad batch {batch_idx+1} truncated '
                    f'(max_tokens=4096, in={response.usage.input_tokens}, '
                    f'out={response.usage.output_tokens}) — increase max_tokens or split batch'
                )

            # strip accidental markdown fences
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[-1]
                raw = raw.rsplit('```', 1)[0]

            results = json.loads(raw)

            for item in results:
                if not item.get('notepad', False):
                    continue
                rid_str   = str(item.get('raceId', ''))
                hname_str = str(item.get('horse', ''))
                hid       = horse_id_lookup.get((rid_str, hname_str))
                if hid is not None:
                    notepad_flags[(rid_str, str(hid))] = True
                    total_flagged += 1

            print(f'  batch {batch_idx+1}/{len(batches)}: {len(results)} results parsed')

        except json.JSONDecodeError as e:
            print(f'  ⚠️  batch {batch_idx+1}: JSON parse error — {e}')
            print(f'      raw response: {raw[:300]}')
        except Exception as e:
            print(f'  ⚠️  batch {batch_idx+1}: API error — {e}')

    print(f'✅ compute_notepad_flags done — {total_flagged} horses flagged across {len(race_ids)} races')
    return notepad_flags


def export_all_races_html(df_hist, df_today,
                          webTips_tdy=None, today_tips=None, races_tdy=None,
                          df_with_ratings=None,
                          odds_min=1.1, odds_max=1.4,
                          notepad_flags=None,
                          pmu_odds_history=None,
                          output_dir=None):
    import os, datetime as _dt

    if output_dir is None:
        output_dir = '.'

    df_today = df_today.copy()
    df_today['_date_dt'] = pd.to_datetime(df_today['date'])
    meetings = df_today.sort_values('_date_dt')['name_meeting'].dropna().unique().tolist()

    def races_for(meeting):
        sub = df_today[df_today['name_meeting'] == meeting]
        _, idx = np.unique(sub['name_race'].dropna().values, return_index=True)
        return sub['name_race'].dropna().iloc[np.sort(idx)].tolist()

    total_races = sum(len(races_for(m)) for m in meetings)
    race_counter = [0]
    today_str = _dt.date.today().strftime('%Y-%m-%d')
    print(f'🏇 Exporting {total_races} races across {len(meetings)} meetings...')

    saved      = []
    race_jsons = {}   # {race_key: race_json} — same data as HTML, built once

    for meeting in meetings:
        print(f'\n📍 {meeting}')
        for race in races_for(meeting):
            race_counter[0] += 1
            pct = int(race_counter[0] / total_races * 20)
            bar = '█' * pct + '░' * (20 - pct)
            print(f'  [{bar}] {race_counter[0]}/{total_races}  {race}', flush=True)

            race_rows = df_today[
                (df_today['name_meeting'] == meeting) &
                (df_today['name_race']    == race)
            ]

            # ── header ───────────────────────────────────────────
            going = going_cat = distance = total_prize = race_type = race_class = age_range_str = ''
            if not race_rows.empty:
                if 'going' in race_rows.columns:
                    v = race_rows['going'].dropna()
                    going = str(v.iloc[0]) if not v.empty else ''
                if 'going_category' in race_rows.columns:
                    v = race_rows['going_category'].dropna()
                    going_cat = str(v.iloc[0]).strip() if not v.empty else ''
                if 'distance' in race_rows.columns:
                    try: distance = f'{float(race_rows["distance"].dropna().iloc[0]):.0f}m'
                    except: pass
                for col in ('type','raceType'):
                    if col in race_rows.columns:
                        v = race_rows[col].dropna()
                        if not v.empty and str(v.iloc[0]).strip():
                            race_type = str(v.iloc[0]).strip(); break
                if 'age' in race_rows.columns:
                    ages = pd.to_numeric(race_rows['age'], errors='coerce').dropna()
                    if not ages.empty:
                        mn, mx = int(ages.min()), int(ages.max())
                        age_range_str = f'{mn}-{mx} year olds' if mn!=mx else f'{mn} year olds'
                if 'sex' in race_rows.columns:
                    sexes = [str(s).strip() for s in race_rows['sex'].dropna().unique() if str(s).strip()]
                    fc = {'F','M','FI','JF','FILLY','MARE','POULICHE','JUMENT'}
                    if sexes and all(s.upper() in fc for s in sexes) and age_range_str:
                        hf  = any(s.upper() in {'F','FI','JF','FILLY','POULICHE'} for s in sexes)
                        hm2 = any(s.upper() in {'M','MARE','JUMENT'} for s in sexes)
                        sfx = 'fillies & mares' if (hf and hm2) else ('fillies' if hf else 'mares')
                        age_range_str += ' ' + sfx

            total_prize_raw = None
            if races_tdy is not None:
                mask = pd.Series(True, index=races_tdy.index)
                if 'name_meeting' in races_tdy.columns: mask &= races_tdy['name_meeting'] == meeting
                if 'name_race'    in races_tdy.columns: mask &= races_tdy['name_race']    == race
                matched = races_tdy[mask]
                if not matched.empty:
                    for pc in ('totalPrize','raceTotalPrize','prize'):
                        if pc in matched.columns:
                            pv = matched[pc].dropna()
                            if not pv.empty:
                                try:
                                    fv = float(pv.iloc[0])
                                    total_prize     = '€'+(f'{fv/1000:.0f}k' if fv>=1000 else f'{fv:.0f}')
                                    total_prize_raw = fv
                                except: pass
                                break
                    for cc in ('class','raceClass'):
                        if cc in matched.columns:
                            cv = matched[cc].dropna()
                            if not cv.empty and str(cv.iloc[0]).strip():
                                race_class = str(cv.iloc[0]).strip(); break

            hparts = [f'<strong>{meeting}</strong>', f'<em>{race}</em>']
            if race_type:     hparts.append(f'<strong>{race_type}</strong>')
            if race_class:    hparts.append(f'Cl: <strong>{race_class}</strong>')
            if total_prize:   hparts.append(f'<strong style="color:#7ecfa0">{total_prize}</strong>')
            if distance:      hparts.append(f'<strong>{distance}</strong>')
            if going:         hparts.append(f'<strong>{going}</strong>')
            if going_cat and going_cat != going:
                              hparts.append(f'<span style="opacity:.75;font-size:12px">[{going_cat}]</span>')
            if age_range_str: hparts.append(f'<span style="opacity:.85">{age_range_str}</span>')

            header_html = (
                f'<div style="background:{TF["navy"]};color:white;padding:10px 16px;border-radius:6px;'
                f'font-family:\'Helvetica Neue\',Arial,sans-serif;font-size:14px;'
                f'display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin-bottom:8px">'
                f'{"&nbsp;·&nbsp;".join(hparts)}</div>'
            )

            # ── Paris Turf verdict ────────────────────────────────
            paristurf_verdict = None
            verdict_html = ''
            if webTips_tdy is not None and 'raceId' in race_rows.columns:
                rids = race_rows['raceId'].dropna().unique()
                tip_rows = webTips_tdy[webTips_tdy['raceId'].isin(rids)]
                if not tip_rows.empty and 'text' in tip_rows.columns:
                    vt = ' '.join(tip_rows['text'].dropna().astype(str)).strip()
                    if vt:
                        paristurf_verdict = vt
                        verdict_html = (
                            f'<div style="background:#fffdf0;border-left:4px solid {TF["green"]};'
                            f'border-radius:0 6px 6px 0;padding:10px 16px;font-size:13px;color:#333;'
                            f'margin-bottom:10px;line-height:1.6;font-family:\'Helvetica Neue\',Arial,sans-serif">'
                            f'<div style="font-size:11px;font-weight:bold;color:{TF["green"]};'
                            f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Paristurf Verdict</div>'
                            f'{vt}</div>'
                        )

            # ── runners (also returns race JSON built from the same data) ──────
            runners_html, _race_json = _render_runners_html(
                race_rows, df_hist,
                today_tips=today_tips,
                df_with_ratings=df_with_ratings,
                odds_min=odds_min, odds_max=odds_max,
                notepad_flags=notepad_flags,
                pmu_odds_history=pmu_odds_history,
            )
            # Augment with race-level fields not available inside _render_runners_html
            _race_json['total_prize_eur']     = total_prize_raw
            _race_json['race_class']          = race_class
            _race_json['paristurf_verdict']   = paristurf_verdict

            # ── assemble page ─────────────────────────────────────
            import re
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', f'{meeting}__{race}')
            filename  = f'{today_str}__{safe_name}.html'
            filepath  = os.path.join(output_dir, filename)
            race_jsons[f'{today_str}__{safe_name}'] = _race_json

            _rv_key = re.sub(r'[^A-Za-z0-9]', '_', f'{today_str}__{safe_name}')
            page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{meeting} — {race}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #f0f3f8; padding: 16px; }}
  ::-webkit-scrollbar {{ height: 5px; width: 5px; }}
  ::-webkit-scrollbar-thumb {{ background: {TF['border']}; border-radius: 3px; }}
</style>
</head>
<body>
{header_html}
{verdict_html}
<!--RACE_VERDICT:{_rv_key}--><!--RACE_VERDICT_END:{_rv_key}-->
{runners_html}
</body>
</html>"""

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_html)
            saved.append(filename)

    print(f'\n✅ Done — {len(saved)} files saved to {output_dir}')
    return saved, race_jsons


# ─────────────────────────────────────────────────────────────────────────────
# Verdict helpers: Claude API call + fast HTML injection
# ─────────────────────────────────────────────────────────────────────────────


def _anthropic_create_with_retry(client, max_retries=3, **kwargs):
    """Call client.messages.create with retry on RateLimitError (waits 60s × attempt)."""
    import anthropic as _anthropic
    import time as _time
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except _anthropic.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = 60 * (attempt + 1)
            print(f'  ⚠️  Rate limit — waiting {wait}s before retry {attempt + 1}/{max_retries} '
                  f'({kwargs.get("model","?")} call)...')
            _time.sleep(wait)


def generate_combined_verdict(race_json, api_key, learnings_db=None, max_learnings=20):
    """
    Single API call that returns both horse-level verdicts and the NAP/EW selection.
    Replaces the two separate generate_race_verdicts + generate_race_verdict calls.

    learnings_db entries are sorted by counter (desc) and the top max_learnings are
    injected as a cached system-prompt block — so all races in one session share the
    cached token cost (Anthropic ephemeral cache, 5-min TTL).

    Returns dict with:
      'verdicts'  — {horse_name: verdict_text, ...}
      'nap'       — {horse, confidence, reason}
      'each_way'  — {horse, confidence, reason}
    or {} on failure.
    """
    import anthropic as _anthropic
    import json as _json
    import re as _re

    client = _anthropic.Anthropic(api_key=api_key)

    # ── Build system blocks with prompt caching ───────────────────────────────
    system_blocks = [
        {
            'type': 'text',
            'text': COMBINED_VERDICT_SYSTEM_PROMPT,
            'cache_control': {'type': 'ephemeral'},
        }
    ]
    if learnings_db:
        top_learnings = sorted(
            learnings_db, key=lambda x: x.get('counter', 0), reverse=True
        )[:max_learnings]
        lines = [f'{i+1}. [n={e.get("counter",1)}] {e.get("learning","")}'
                 for i, e in enumerate(top_learnings)]
        learnings_block = (
            f'## Past Learnings (top {len(top_learnings)} by confirmation count)\n'
            + '\n'.join(lines)
        )
        system_blocks.append({
            'type': 'text',
            'text': learnings_block,
            'cache_control': {'type': 'ephemeral'},
        })

    # ── User message — race data only (no learnings blob) ────────────────────
    horse_names = [h['name'] for h in race_json.get('horses', [])]
    user_msg = (
        f'Write a verdict for each horse and select the NAP and EACH WAY.\n'
        f'Horses: {_json.dumps(horse_names)}\n\n'
        f'Race data:\n{_json.dumps(race_json, indent=2, default=str)}'
    )

    resp = _anthropic_create_with_retry(
        client,
        model='claude-sonnet-4-6',
        max_tokens=4096,
        system=system_blocks,
        messages=[{'role': 'user', 'content': user_msg}],
    )

    _race_label = race_json.get('race', race_json.get('meeting', '?'))
    _cache_read = getattr(resp.usage, 'cache_read_input_tokens', 0) or 0
    _cache_created = getattr(resp.usage, 'cache_creation_input_tokens', 0) or 0
    _cache_info = (f'  💾 cache hit {_cache_read:,} tok' if _cache_read
                   else f'  📝 cache write {_cache_created:,} tok' if _cache_created
                   else '')
    print(f'  {_race_label}: in={resp.usage.input_tokens} out={resp.usage.output_tokens}{_cache_info}')
    if resp.stop_reason == 'max_tokens':
        print(
            f'  ⚠️  TOKEN LIMIT: generate_combined_verdict "{_race_label}" truncated '
            f'(max_tokens=4096, in={resp.usage.input_tokens}, '
            f'out={resp.usage.output_tokens}) — increase max_tokens'
        )

    text = resp.content[0].text.strip()
    text = _re.sub(r'^```(?:json)?\s*', '', text)
    text = _re.sub(r'\s*```$', '', text)
    match = _re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            result = _json.loads(match.group())
            if 'nap' in result and 'each_way' in result:
                return result
        except _json.JSONDecodeError:
            pass
    return {}


def _render_race_verdict_html(verdict):
    """Render NAP/EW verdict dict as an HTML block for injection into race page HTML."""
    if not verdict or 'nap' not in verdict:
        return ''
    nap = verdict.get('nap', {})
    ew  = verdict.get('each_way', {})

    def _badge(label, color):
        return (
            f'<span style="display:inline-block;padding:1px 7px;border-radius:10px;'
            f'font-size:10px;font-weight:bold;background:{color};color:#fff;'
            f'letter-spacing:.05em;margin-right:6px">{label}</span>'
        )

    def _row(label, color, data):
        horse      = data.get('horse', '')
        confidence = data.get('confidence', '')
        reason     = data.get('reason', '')
        conf_color = '#2d7d2d' if int(confidence or 0) >= 7 else ('#b8860b' if int(confidence or 0) >= 5 else '#c0392b')
        return (
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px">'
            f'{_badge(label, color)}'
            f'<div>'
            f'<span style="font-weight:bold;color:#1a1a2e;font-size:13px">{horse}</span>'
            f'<span style="font-size:11px;color:{conf_color};margin-left:8px">'
            f'(Confidence: {confidence}/10)</span>'
            f'<div style="font-size:12px;color:#444;margin-top:2px">{reason}</div>'
            f'</div>'
            f'</div>'
        )

    html = (
        '<div style="background:#fff8f0;border-left:4px solid #c0392b;'
        'border-radius:0 6px 6px 0;padding:10px 16px;font-size:13px;'
        'margin-bottom:10px;line-height:1.6;'
        'font-family:\'Helvetica Neue\',Arial,sans-serif">'
        '<div style="font-size:11px;font-weight:bold;color:#c0392b;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Race Verdict</div>'
        + _row('NAP', '#c0392b', nap)
        + _row('EW',  '#8e44ad', ew)
        + '</div>'
    )
    return html


def _render_runners_html(race_rows, runners_hist,
                         today_tips=None,
                         df_with_ratings=None,
                         odds_min=1.1, odds_max=1.4, notepad_flags=None, pmu_odds_history=None,
                         horse_verdicts=None,):
    """
    Standalone HTML renderer for a race's runners — used by export_all_races_html.
    Mirrors the logic of _render_runners() in display_race(), including the ARR row.
    """
    import datetime as _dt
    import bisect as _bisect
    from scipy import stats as _scipy_stats

    if race_rows.empty:
        return ''

    rows = race_rows.copy()

    horse_odds_strip = {}
    if pmu_odds_history is not None and 'horseName' in pmu_odds_history.columns:
        _oh = pmu_odds_history.copy()
        _oh['_ts'] = pd.to_datetime(_oh['timestamp'], errors='coerce')
        _oh = _oh.dropna(subset=['_ts', 'odds', 'horseName'])
        _oh = _oh.sort_values('_ts', ascending=True)

        for _hname, _grp in _oh.groupby('horseName'):
            _odds_series = _grp['odds'].tolist()
            _ts_series   = _grp['_ts'].tolist()
            if len(_odds_series) == 0:
                continue

            # deduplicate consecutive identical values
            _deduped_idx = [0]
            for _i in range(1, len(_odds_series) - 1):
                if _odds_series[_i] != _odds_series[_i - 1]:
                    _deduped_idx.append(_i)
            if len(_odds_series) > 1:
                _deduped_idx.append(len(_odds_series) - 1)
            _deduped_idx = sorted(set(_deduped_idx))

            _deduped_odds = [_odds_series[i] for i in _deduped_idx]
            _deduped_ts   = [_ts_series[i]   for i in _deduped_idx]

            # pick first + last + up to 3 interior via quantiles
            if len(_deduped_odds) <= 5:
                _picked = list(range(len(_deduped_odds)))
            else:
                _interior_idx = list(range(1, len(_deduped_odds) - 1))
                _q = np.quantile(_interior_idx, [0.25, 0.5, 0.75])
                _q_positions = sorted(set(int(round(q)) for q in _q))
                _picked = sorted(set([0] + _q_positions + [len(_deduped_odds) - 1]))

            horse_odds_strip[str(_hname)] = [
                (_deduped_ts[i], _deduped_odds[i]) for i in _picked
            ]

    if today_tips is not None and 'raceId' in rows.columns and 'horse' in today_tips.columns:
        tips_sub = today_tips[['raceId', 'horse', 'SP']].rename(columns={'horse': 'horseName'})
        rows = rows.merge(tips_sub, on=['raceId', 'horseName'], how='left')

    if (df_with_ratings is not None
            and 'horseId' in rows.columns
            and 'horseId' in df_with_ratings.columns
            and 'rating_after_race' in df_with_ratings.columns):

        # Get the most recent rating_after_race per horse
        _rat_df = df_with_ratings[['horseId', 'rating_after_race']].copy()

        if 'date' in df_with_ratings.columns:
            _rat_df['_rtr_date'] = pd.to_datetime(df_with_ratings['date'], errors='coerce')
            _rat_df = (_rat_df
                      .sort_values('_rtr_date', ascending=True)
                      .drop_duplicates(subset=['horseId'], keep='last')
                      .drop(columns=['_rtr_date']))
        else:
            _rat_df = _rat_df.drop_duplicates(subset=['horseId'], keep='last')

        if 'rating_after_race' in rows.columns:
            rows = rows.drop(columns=['rating_after_race'])
        rows = rows.merge(_rat_df, on='horseId', how='left')

    if 'SP' in rows.columns:
        rows = rows.sort_values('SP', ascending=True, na_position='last')

    today_str  = _dt.datetime.today().strftime('%Y-%m-%d')
    cutoff_365 = (_dt.datetime.today() - _dt.timedelta(days=365)).strftime('%Y-%m-%d')
    cutoff_750 = (_dt.datetime.today() - _dt.timedelta(days=750)).strftime('%Y-%m-%d')
    cutoff_21  = (_dt.datetime.today() - _dt.timedelta(days=21)).strftime('%Y-%m-%d')

    c_border = TF['border']
    c_navy   = TF['navy']
    c_green  = TF['green']

    # ── CHANGE 4 & 5: Split stats into AE (odds-filtered) and base (no odds filter) ──

    def _compute_ae_stats(hist_df, grouper_col, cutoff_date=None):
        """A/E stats ONLY — applies the odds_min/odds_max filter."""
        out = {}
        if hist_df is None or grouper_col not in hist_df.columns:
            return out
        h = hist_df.copy()
        if 'liveOdd' in h.columns and 'odds_sum' in h.columns and 'date' in h.columns:
            h['_dt2'] = pd.to_datetime(h['date'], errors='coerce')
            mask = (
                h['liveOdd'].notnull() &
                h['odds_sum'].between(odds_min, odds_max) &
                (h['_dt2'] < today_str)
            )
            if cutoff_date is not None:
                mask = mask & (h['_dt2'] >= cutoff_date)
            h = h.loc[mask]
        if h.empty or 'win' not in h.columns:
            return out
        for name, grp in h.groupby(grouper_col):
            if pd.isna(name):
                continue
            places = int(grp['place'].sum()) if 'place' in grp.columns else 0
            ae_place = None
            if 'place' in grp.columns and 'odds_place' in grp.columns:
                denom = grp['odds_place'].sum()
                ae_place = round(places / denom, 2) if denom > 0 else None
            out[str(name)] = dict(ae_place=ae_place)
        return out

    def _compute_base_stats(hist_df, grouper_col, cutoff_date=None):
        """R/W/P and €/R stats — NO odds filter."""
        out = {}
        if hist_df is None or grouper_col not in hist_df.columns:
            return out
        h = hist_df.copy()
        if 'date' in h.columns:
            h['_dt2'] = pd.to_datetime(h['date'], errors='coerce')
            mask = h['_dt2'] < today_str
            if cutoff_date is not None:
                mask = mask & (h['_dt2'] >= cutoff_date)
            h = h.loc[mask]
        if h.empty or 'win' not in h.columns:
            return out
        for name, grp in h.groupby(grouper_col):
            if pd.isna(name):
                continue
            runs   = len(grp)
            wins   = int(grp['win'].sum())
            places = int(grp['place'].sum()) if 'place' in grp.columns else 0
            prizemoney = None
            if 'prizemoney' in grp.columns and runs > 0:
                prizemoney = round(grp['prizemoney'].sum() / runs, 0)
            out[str(name)] = dict(runs=runs, wins=wins, places=places, prizemoney=prizemoney)
        return out

    def _compute_base_stats_sire_median(hist_df, grouper_col, cutoff_date=None):
        """Like _compute_base_stats but uses MEDIAN prizemoney — for sire €/R."""
        out = {}
        if hist_df is None or grouper_col not in hist_df.columns:
            return out
        h = hist_df.copy()
        if 'date' in h.columns:
            h['_dt2'] = pd.to_datetime(h['date'], errors='coerce')
            mask = h['_dt2'] < today_str
            if cutoff_date is not None:
                mask = mask & (h['_dt2'] >= cutoff_date)
            h = h.loc[mask]
        if h.empty or 'win' not in h.columns:
            return out
        for name, grp in h.groupby(grouper_col):
            if pd.isna(name):
                continue
            runs   = len(grp)
            wins   = int(grp['win'].sum())
            places = int(grp['place'].sum()) if 'place' in grp.columns else 0
            prizemoney = None
            if 'prizemoney' in grp.columns and runs > 0:
                pm_vals = pd.to_numeric(grp['prizemoney'], errors='coerce').dropna()
                if not pm_vals.empty:
                    prizemoney = round(float(pm_vals.mean()), 0)
            out[str(name)] = dict(runs=runs, wins=wins, places=places, prizemoney=prizemoney)
        return out

    # Base stats (no odds filter) for R/W/P and €/R
    base_by_horse   = _compute_base_stats(runners_hist, 'horseId')
    base_by_sire    = _compute_base_stats_sire_median(runners_hist, 'horseSir')  # median for sire
    base_by_trainer = _compute_base_stats(runners_hist, 'trainerName', cutoff_date=cutoff_365)
    base_by_jockey  = _compute_base_stats(runners_hist, 'jockeyName',  cutoff_date=cutoff_365)

    def _compute_horse_condition_stats(hist_df):
        """For each horse compute stats split by distance_group and going_category."""
        out = {}
        if hist_df is None or 'horseId' not in hist_df.columns:
            return out
        h = hist_df.copy()
        if 'date' in h.columns:
            h['_dt2'] = pd.to_datetime(h['date'], errors='coerce')
            h = h.loc[h['_dt2'] < today_str]
        if h.empty or 'win' not in h.columns:
            return out
        for hid2, grp in h.groupby('horseId'):
            entry = {}
            # distance_group breakdown
            if 'distance_group' in grp.columns:
                dist_stats = {}
                for dg, dgrp in grp.groupby('distance_group'):
                    if pd.isna(dg) or str(dg).strip() in ('', 'nan'):
                        continue
                    r = len(dgrp)
                    w = int(dgrp['win'].sum())
                    p = int(dgrp['place'].sum()) if 'place' in dgrp.columns else 0
                    dist_stats[str(dg)] = {'runs': r, 'wins': w, 'places': p}
                entry['by_distance'] = dist_stats
            # going_category breakdown
            if 'going_category' in grp.columns:
                going_stats = {}
                for gc, ggrp in grp.groupby('going_category'):
                    if pd.isna(gc) or str(gc).strip() in ('', 'nan'):
                        continue
                    r = len(ggrp)
                    w = int(ggrp['win'].sum())
                    p = int(ggrp['place'].sum()) if 'place' in ggrp.columns else 0
                    going_stats[str(gc)] = {'runs': r, 'wins': w, 'places': p}
                entry['by_going'] = going_stats
            if entry:
                out[str(int(hid2)) if not pd.isna(hid2) else str(hid2)] = entry
        return out

    horse_condition_stats = _compute_horse_condition_stats(runners_hist)

    # A/E stats (odds filter) for ae_place only
    ae_by_horse   = _compute_ae_stats(runners_hist, 'horseId')
    ae_by_sire    = _compute_ae_stats(runners_hist, 'horseSir')
    ae_by_trainer = _compute_ae_stats(runners_hist, 'trainerName', cutoff_date=cutoff_365)
    ae_by_jockey  = _compute_ae_stats(runners_hist, 'jockeyName',  cutoff_date=cutoff_365)

    def _merge_stats(base_dict, ae_dict):
        merged = {}
        all_keys = set(base_dict.keys()) | set(ae_dict.keys())
        for k in all_keys:
            entry = dict(base_dict.get(k, {}))
            entry['ae_place'] = ae_dict.get(k, {}).get('ae_place', None)
            merged[k] = entry
        return merged

    stats_by_horse   = _merge_stats(base_by_horse,   ae_by_horse)
    stats_by_sire    = _merge_stats(base_by_sire,     ae_by_sire)
    stats_by_trainer = _merge_stats(base_by_trainer,  ae_by_trainer)
    stats_by_jockey  = _merge_stats(base_by_jockey,   ae_by_jockey)

    def _compute_hot_cold(hist_df, grouper_col):
        out = {}
        if hist_df is None or grouper_col not in hist_df.columns or 'pos_perc' not in hist_df.columns:
            return out
        h = hist_df.copy()
        if 'date' not in h.columns:
            return out
        h['_hcdt'] = pd.to_datetime(h['date'], errors='coerce')
        mask = (h['_hcdt'] >= cutoff_21) & (h['_hcdt'] < today_str)
        h = h.loc[mask].dropna(subset=['pos_perc'])
        if h.empty:
            return out
        for name, grp in h.groupby(grouper_col):
            if pd.isna(name):
                continue
            runs   = len(grp)
            wins   = int(grp['win'].sum()) if 'win' in grp.columns else 0
            places = int(grp['place'].sum()) if 'place' in grp.columns else 0
            avg_pp = round(float(grp['pos_perc'].mean()), 3)
            out[str(name)] = dict(avg_pos_perc=avg_pp, runs=runs, wins=wins, places=places)
        return out

    hot_cold_trainer = _compute_hot_cold(runners_hist, 'trainerName')
    hot_cold_jockey  = _compute_hot_cold(runners_hist, 'jockeyName')

    def _compute_avg_pp_365(hist_df, grouper_col):
        """Average pos_perc over last 365 days per entity."""
        out = {}
        if hist_df is None or grouper_col not in hist_df.columns or 'pos_perc' not in hist_df.columns:
            return out
        h = hist_df.copy()
        if 'date' not in h.columns:
            return out
        h['_ppdt'] = pd.to_datetime(h['date'], errors='coerce')
        mask = (h['_ppdt'] >= cutoff_365) & (h['_ppdt'] < today_str)
        h = h.loc[mask].dropna(subset=['pos_perc'])
        if h.empty:
            return out
        for name, grp in h.groupby(grouper_col):
            if pd.isna(name):
                continue
            out[str(name)] = round(float(grp['pos_perc'].mean()), 3)
        return out

    avg_pp_365_trainer = _compute_avg_pp_365(runners_hist, 'trainerName')
    avg_pp_365_jockey  = _compute_avg_pp_365(runners_hist, 'jockeyName')
    avg_pp_365_sire    = _compute_avg_pp_365(runners_hist, 'horseSir')
    avg_pp_365_owner   = _compute_avg_pp_365(runners_hist, 'ownerName')

    # collect all values for percentile colouring
    all_pp365_trainer = list(avg_pp_365_trainer.values())
    all_pp365_jockey  = list(avg_pp_365_jockey.values())
    all_pp365_sire    = list(avg_pp_365_sire.values())

    draw_pos_perc = {}
    if runners_hist is not None and 'draw_unique' in runners_hist.columns and 'pos_perc' in runners_hist.columns:
        try:
            _dh = runners_hist.copy()
            if 'date' in _dh.columns:
                _dh['_ddt'] = pd.to_datetime(_dh['date'], errors='coerce')
                _dh = _dh[_dh['_ddt'] < today_str]
            _dh = _dh.dropna(subset=['pos_perc', 'draw_unique'])
            for du, grp in _dh.groupby('draw_unique'):
                n = len(grp)
                avg_pp = float(grp['pos_perc'].mean()) - 0.5
                draw_pos_perc[str(du)] = (round(avg_pp, 3), n)
        except Exception:
            pass

    cat_ae  = {'horse': [], 'trainer': [], 'jockey': [], 'sire': []}
    cat_pm  = {'horse': [], 'trainer': [], 'jockey': [], 'sire': []}
    cat_rtr = []
    cat_val = []

    for _, r in rows.iterrows():
        hid     = (str(int(r['horseId'])) if 'horseId' in r.index and pd.notna(r.get('horseId')) else '')
        trainer = r.get('trainerName', ''); trainer = str(trainer) if pd.notna(trainer) else ''
        jockey  = r.get('jockeyName',  ''); jockey  = str(jockey)  if pd.notna(jockey)  else ''
        sire    = r.get('horseSir',    ''); sire    = str(sire)    if pd.notna(sire)    else ''
        for cat, st in [('horse',   stats_by_horse.get(hid, {})),
                        ('trainer', stats_by_trainer.get(trainer, {})),
                        ('jockey',  stats_by_jockey.get(jockey,   {})),
                        ('sire',    stats_by_sire.get(sire,        {}))]:
            if st.get('ae_place') is not None:
                cat_ae[cat].append(st['ae_place'])
            if st.get('prizemoney') is not None:
                cat_pm[cat].append(st['prizemoney'])
        rtr_val = r.get('rating_after_race', None)
        if rtr_val is not None and pd.notna(rtr_val):
            try:
                _rtr_raw = float(rtr_val)
                _wt_v = r.get('weightKg', None)
                if _wt_v is not None and pd.notna(_wt_v):
                    cat_rtr.append(_rtr_raw - float(_wt_v) + 55.0)
                else:
                    cat_rtr.append(_rtr_raw)
            except: pass
        try:
            hcap_v = float(r.get('handicapRatingKg', None))
            wt_v   = float(r.get('weightKg', None))
            if pd.notna(hcap_v) and pd.notna(wt_v):
                cat_val.append(hcap_v - wt_v + 55)
        except Exception:
            pass

    def _prep_hist_for_prefs(hist_df, cutoff_str):
        """Prepare hist for preference t-tests — NO odds filter."""
        if hist_df is None or 'pos_perc' not in hist_df.columns:
            return None
        h = hist_df.copy()
        if 'date' not in h.columns:
            return h
        h['_pdt'] = pd.to_datetime(h['date'], errors='coerce')
        mask = (h['_pdt'] >= cutoff_str) & (h['_pdt'] < today_str)
        return h.loc[mask].copy()

    _hist_750 = _prep_hist_for_prefs(runners_hist, cutoff_750)
    _hist_all = _prep_hist_for_prefs(runners_hist, '1900-01-01')

    def _ttest_pref(hist_df, entity_col, entity_val, subgroup_col, today_condition_val):
        if hist_df is None or entity_col not in hist_df.columns or subgroup_col not in hist_df.columns:
            return None, None, 0
        if today_condition_val is None or (isinstance(today_condition_val, float) and pd.isna(today_condition_val)):
            return None, None, 0
        cond_str = str(today_condition_val)
        ent = hist_df[hist_df[entity_col].astype(str) == str(entity_val)]
        if ent.empty:
            return None, None, 0
        ent = ent.dropna(subset=['pos_perc'])
        mask_in  = ent[subgroup_col].astype(str) == cond_str
        in_vals  = ent.loc[mask_in,  'pos_perc'].values
        out_vals = ent.loc[~mask_in, 'pos_perc'].values
        if len(in_vals) < 3 or len(out_vals) < 3:
            return None, None, len(in_vals)
        try:
            t_stat, p_val = _scipy_stats.ttest_ind(in_vals, out_vals, equal_var=False)
            return round(float(t_stat), 2), round(float(p_val), 4), len(in_vals)
        except Exception:
            return None, None, len(in_vals)

    def _pref_chip_html(icon, label, t_stat, p_val, n_in, today_val):
        if t_stat is None:
            return ''
        t_col  = '#1a7a3a' if t_stat > 0 else '#c0392b'
        t_bg   = '#d4edda' if t_stat > 0 else '#fde8e8'
        t_bdr  = '#a8d5b5' if t_stat > 0 else '#f5b8b8'
        sig    = '★' if p_val is not None and p_val < 0.05 else ''
        sig_col = '#e67e22' if sig else ''
        disp_val = str(today_val)
        if len(disp_val) > 14:
            disp_val = disp_val[:13] + '…'
        return (
            '<span style="display:inline-flex;align-items:center;gap:2px;'
            'background:' + t_bg + ';border-radius:8px;padding:1px 6px;'
            'font-size:10px;border:1px solid ' + t_bdr + ';white-space:nowrap" '
            'title="' + icon + ' ' + label + ': ' + str(today_val) + ' | t=' + str(t_stat) + ' p=' + str(p_val) + ' n=' + str(n_in) + '">'
            + icon +
            '<span style="color:#666;margin-left:1px">' + disp_val + '</span>'
            '<strong style="color:' + t_col + ';margin-left:2px">' + str(t_stat) + '</strong>'
            + (('<span style="color:' + sig_col + ';margin-left:1px;font-size:9px">' + sig + '</span>') if sig else '')
            + '</span>'
        )

    # Pre-group history DFs by entity to make _ttest_pref O(1) lookup instead of O(N) scan
    def _pregroup(hist_df, col):
        if hist_df is None or col not in hist_df.columns:
            return {}
        return {str(k): grp for k, grp in hist_df.groupby(col)}

    _hist_750_by_trainer = _pregroup(_hist_750, 'trainerName')
    _hist_750_by_jockey  = _pregroup(_hist_750, 'jockeyName')
    _hist_all_by_horse   = _pregroup(_hist_all, 'horseId')
    _hist_all_by_sire    = _pregroup(_hist_all, 'horseSir')

    def _ttest_from_group(grouped, entity_val, subgroup_col, today_condition_val):
        """Like _ttest_pref but uses pre-grouped dict — O(1) lookup."""
        if today_condition_val is None or (isinstance(today_condition_val, float) and pd.isna(today_condition_val)):
            return None, None, 0
        ent = grouped.get(str(entity_val))
        if ent is None or ent.empty:
            return None, None, 0
        if subgroup_col not in ent.columns or 'pos_perc' not in ent.columns:
            return None, None, 0
        ent = ent.dropna(subset=['pos_perc'])
        cond_str = str(today_condition_val)
        mask_in  = ent[subgroup_col].astype(str) == cond_str
        in_vals  = ent.loc[mask_in,  'pos_perc'].values
        out_vals = ent.loc[~mask_in, 'pos_perc'].values
        if len(in_vals) < 3 or len(out_vals) < 3:
            return None, None, len(in_vals)
        try:
            t_stat, p_val = _scipy_stats.ttest_ind(in_vals, out_vals, equal_var=False)
            return round(float(t_stat), 2), round(float(p_val), 4), len(in_vals)
        except Exception:
            return None, None, len(in_vals)

    def _build_prefs_data(entity_col, entity_val, hist_df, specs):
        """Compute t-test preferences once — returns list of dicts for JSON and HTML."""
        if entity_val == '—' or not entity_val:
            return []
        # Use pre-grouped dicts for O(1) entity lookup; fall back to full DF scan
        _grouped_map = {
            ('trainerName', id(_hist_750)): _hist_750_by_trainer,
            ('jockeyName',  id(_hist_750)): _hist_750_by_jockey,
            ('horseId',     id(_hist_all)): _hist_all_by_horse,
            ('horseSir',    id(_hist_all)): _hist_all_by_sire,
        }
        _grouped = _grouped_map.get((entity_col, id(hist_df)))
        result = []
        for icon, label, subgroup_col, today_val in specs:
            if _grouped is not None:
                t, p, n = _ttest_from_group(_grouped, entity_val, subgroup_col, today_val)
            else:
                t, p, n = _ttest_pref(hist_df, entity_col, entity_val, subgroup_col, today_val)
            if t is not None:
                result.append({
                    'icon': icon, 'label': label, 'condition': today_val,
                    't': t, 'p': p, 'n': n, 'sig': (p is not None and p < 0.05),
                })
        return result

    def _prefs_data_to_html(prefs_data):
        """Render pref chips from pre-computed _build_prefs_data result — no t-tests."""
        if not prefs_data:
            return ''
        chips = [
            _pref_chip_html(item['icon'], item['label'], item['t'], item['p'], item['n'], item['condition'])
            for item in prefs_data
        ]
        chips = [c for c in chips if c]
        if not chips:
            return ''
        return (
            '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:3px">'
            + ''.join(chips)
            + '</div>'
        )

    def _today_val(col):
        if col in rows.columns:
            v = rows[col].dropna()
            if not v.empty:
                return str(v.iloc[0])
        return None

    today_name_meeting = _today_val('name_meeting')
    today_distance_grp = _today_val('distance_group')
    today_going_grp    = _today_val('going_category')
    today_racetype     = _today_val('type') or _today_val('raceType')

    # ── per-horse today weight lookup (for ARR and VAL adjustment) ────────────
    horse_today_weight = {}
    if 'horseId' in rows.columns and 'weightKg' in rows.columns:
        for _, r_wt in rows.iterrows():
            hid_wt = r_wt.get('horseId')
            wt_wt  = r_wt.get('weightKg')
            if pd.notna(hid_wt) and pd.notna(wt_wt):
                try:
                    horse_today_weight[str(int(hid_wt))] = float(wt_wt)
                except Exception:
                    pass

    # ── today's starter lookup: horseId -> horseName  (for Opp row) ──────────
    today_starter_names = {}
    if 'horseId' in rows.columns and 'horseName' in rows.columns:
        for _, r_sn in rows.iterrows():
            hid_sn = r_sn.get('horseId')
            hn_sn  = r_sn.get('horseName')
            if pd.notna(hid_sn) and pd.notna(hn_sn):
                try:
                    today_starter_names[str(int(hid_sn))] = str(hn_sn)
                except Exception:
                    pass

    # ── today's starter SP lookup: horseId -> SP  (for Opp2 sorting) ─────────
    today_starter_sp = {}
    if 'horseId' in rows.columns and 'SP' in rows.columns:
        for _, r_sp in rows.iterrows():
            hid_sp = r_sp.get('horseId')
            sp_sp  = r_sp.get('SP')
            if pd.notna(hid_sp) and pd.notna(sp_sp):
                try:
                    today_starter_sp[str(int(hid_sp))] = float(sp_sp)
                except Exception:
                    pass

    horse_last_start   = {}
    horse_form_context = {}

    if runners_hist is not None and 'horseId' in rows.columns:
        horse_ids = rows['horseId'].dropna().unique().tolist()

        _needed_cols = ['horseId', 'raceId', 'prizemoney', 'name_meeting',
                        'going_category', 'distance_group', 'ranking', 'runners',
                        'cumulative_lengths_back', 'handicapRatingKg', 'weightKg',
                        'position', 'pos', 'date', 'ARR', 'rating_after_race',
                        'meetingName', 'type', 'class', 'totalPrize_y', 'distance',
                        'horseName', 'liveOdd', 'jockeyName', 'draw_unique', 'draw',
                        'pos_perc']
        _keep = [c for c in _needed_cols if c in runners_hist.columns]
        _all_hist = runners_hist[_keep].copy()
        _all_hist['_fdt'] = pd.to_datetime(_all_hist['date'], errors='coerce')
        # CHANGE 4/5: No odds filter on form table history
        _all_hist = _all_hist[_all_hist['_fdt'] < today_str]

        if 'prizemoney' in _all_hist.columns and 'raceId' in _all_hist.columns:
            _pm_col = pd.to_numeric(_all_hist['prizemoney'], errors='coerce')
            _all_hist['_pm'] = _pm_col
            _horse_pm_sorted = {}
            for _hid2, _grp in _all_hist.dropna(subset=['_pm']).groupby('horseId'):
                _g = _grp.sort_values('_fdt')
                _ts = _g['_fdt'].values.astype('int64')
                _pm = _g['_pm'].values
                _horse_pm_sorted[_hid2] = (_ts, _pm)
            _365d_ns = int(365 * 24 * 3600 * 1e9)

            def _horse_avg_pm_before(hid2, race_ts_ns):
                entry = _horse_pm_sorted.get(hid2)
                if entry is None:
                    return None
                ts_arr, pm_arr = entry
                lo = _bisect.bisect_left(ts_arr, race_ts_ns - _365d_ns)
                hi = _bisect.bisect_left(ts_arr, race_ts_ns)
                if hi <= lo:
                    return None
                return float(pm_arr[lo:hi].mean())

            _rival_avg_pm = {}
            if 'raceId' in _all_hist.columns:
                _race_dates  = (_all_hist.dropna(subset=['raceId'])
                                .groupby('raceId')['_fdt'].first().to_dict())
                _race_horses = (_all_hist.dropna(subset=['raceId'])
                                .groupby('raceId')['horseId']
                                .apply(lambda x: x.dropna().unique().tolist()).to_dict())
                _today_race_ids = set(
                    _all_hist[_all_hist['horseId'].isin(horse_ids)]['raceId'].dropna().unique()
                )
                for _rid in _today_race_ids:
                    _rdate = _race_dates.get(_rid)
                    if _rdate is None or pd.isna(_rdate):
                        continue
                    _rts = int(pd.Timestamp(_rdate).value)
                    _participants = _race_horses.get(_rid, [])
                    _pm_by_horse = {_p: _horse_avg_pm_before(_p, _rts) for _p in _participants}
                    for _p in _participants:
                        _others = [v for _q, v in _pm_by_horse.items() if _q != _p and v is not None]
                        _rival_avg_pm[(_rid, _p)] = (sum(_others) / len(_others) if _others else None)
        else:
            _rival_avg_pm = {}

        # ── FF: average ARR delta for field members who ran again after each race ──
        ff_stats_by_race = {}
        if 'raceId' in _all_hist.columns and 'ARR' in _all_hist.columns:
            try:
                _arr_hist = _all_hist.dropna(subset=['raceId', 'ARR', 'horseId']).copy()
                _arr_hist['ARR'] = pd.to_numeric(_arr_hist['ARR'], errors='coerce')
                _arr_hist = _arr_hist.dropna(subset=['ARR'])
                # Sort ascending so next-run lookup works
                _arr_hist_sorted = _arr_hist.sort_values('_fdt', ascending=True)

                # Build per-horse sorted ARR history: list of (date, raceId, ARR)
                _horse_arr_history = {
                    hid: list(zip(grp['_fdt'], grp['raceId'], grp['ARR'].astype(float)))
                    for hid, grp in _arr_hist_sorted.groupby('horseId')
                }

                # For each race of interest, compute FF
                _today_race_ids_for_ff = set(
                    _all_hist[_all_hist['horseId'].isin(horse_ids)]['raceId'].dropna().unique()
                )
                for _rid in _today_race_ids_for_ff:
                    _participants_in_race = (
                        _arr_hist[_arr_hist['raceId'] == _rid][['horseId', 'ARR', '_fdt']]
                        .drop_duplicates(subset=['horseId'])
                    )
                    if _participants_in_race.empty:
                        continue
                    deltas = []
                    for _, _p_row in _participants_in_race.iterrows():
                        _p_hid   = _p_row['horseId']
                        _p_arr   = float(_p_row['ARR'])
                        _p_date  = _p_row['_fdt']
                        _p_runs  = _horse_arr_history.get(_p_hid, [])
                        # Find the first run after this race date
                        _next_arr = None
                        for (_run_dt, _run_rid, _run_arr) in _p_runs:
                            if _run_dt > _p_date and _run_rid != _rid:
                                _next_arr = _run_arr
                                break
                        if _next_arr is not None:
                            deltas.append(_next_arr - _p_arr)
                    if deltas:
                        ff_stats_by_race[_rid] = {
                            'avg_delta': round(sum(deltas) / len(deltas), 2),
                            'n': len(deltas),
                        }
            except Exception:
                pass

        _hist_by_horse = {
            hid2: grp.sort_values('_fdt', ascending=False)
            for hid2, grp in _all_hist.groupby('horseId')
        }

        # ── jockey avg pos_perc last 365d (for Jcky row in form table) ──────
        _jockey_avg_pp_365 = {}
        if 'jockeyName' in _all_hist.columns and 'pos_perc' in _all_hist.columns:
            try:
                _jpp = _all_hist.copy()
                _jpp = _jpp[_jpp['_fdt'] >= cutoff_365].dropna(subset=['pos_perc', 'jockeyName'])
                for _jname, _jgrp in _jpp.groupby('jockeyName'):
                    if _jname and str(_jname).strip():
                        _jockey_avg_pp_365[str(_jname)] = round(float(_jgrp['pos_perc'].mean()), 3)
            except Exception:
                pass

        def _build_info_str(race_row):
            """Build a 3-line info string for the Info row:
            Line 1: meetingName
            Line 2: type - class - €Xk   (only non-missing parts joined)
            Line 3: going_category - Xm  (only non-missing parts joined)"""
            def _safe(col):
                v = race_row.get(col, None)
                if v is not None and pd.notna(v) and str(v).strip() and str(v).strip() != 'nan':
                    return str(v).strip()
                return None

            line1 = _safe('meetingName') or '—'

            # totalPrize_y: cast to float, divide by 1000, format as €Xk
            tp = race_row.get('totalPrize_y', None)
            tp_str = None
            if tp is not None and pd.notna(tp):
                try:
                    tp_str = f'€{float(tp) / 1000:.0f}k'
                except (ValueError, TypeError):
                    tp_str = None

            line2_parts = [p for p in [_safe('type'), _safe('class'), tp_str] if p is not None]
            line2 = ' - '.join(line2_parts) if line2_parts else '—'

            # distance: format as Xm
            dist = race_row.get('distance', None)
            dist_str = None
            if dist is not None and pd.notna(dist):
                try:
                    dist_str = f'{float(dist):.0f}m'
                except (ValueError, TypeError):
                    dist_str = None

            line3_parts = [p for p in [_safe('going_category'), dist_str] if p is not None]
            line3 = ' - '.join(line3_parts) if line3_parts else '—'

            return f'{line1}<br>{line2}<br>{line3}'

        # Pre-build race→participants dict so OPP lookup is O(1) not O(N_hist)
        _race_field_cache = {}
        if 'raceId' in _all_hist.columns:
            for _rc_id, _rc_grp in _all_hist.groupby('raceId'):
                _race_field_cache[_rc_id] = _rc_grp

        for hid_raw in horse_ids:
            hid_str    = str(int(hid_raw))
            horse_runs = _hist_by_horse.get(hid_raw, pd.DataFrame())
            if horse_runs.empty:
                continue

            last_dt = horse_runs['_fdt'].iloc[0]
            if pd.notna(last_dt):
                horse_last_start[hid_str] = (_dt.datetime.today() - last_dt).days

            # today's weight for ARR + VAL adjustment
            today_wt = horse_today_weight.get(hid_str, None)

            if 'raceId' in _all_hist.columns and 'prizemoney' in _all_hist.columns:
                ctx_entries = []
                for _, race_row in horse_runs.head(10).iterrows():
                    race_date    = race_row['_fdt']
                    race_id      = race_row.get('raceId', None)
                    run_date_str = race_date.strftime('%b %y') if pd.notna(race_date) else '—'

                    flag_parts = []
                    run_meeting  = str(race_row.get('name_meeting', '') or '').strip()
                    run_going    = str(race_row.get('going_category', '') or '').strip()
                    run_dist_grp = str(race_row.get('distance_group', '') or '').strip()
                    same_meeting = (today_name_meeting and run_meeting and run_meeting == today_name_meeting)
                    same_going   = (today_going_grp    and run_going   and run_going   == today_going_grp)
                    same_dist    = (today_distance_grp and run_dist_grp and run_dist_grp == today_distance_grp)
                    if same_meeting: flag_parts.append('C')
                    if same_going:   flag_parts.append('G')
                    if same_dist:    flag_parts.append('D')
                    cond_flag = ''.join(flag_parts)

                    run_pos = None
                    for pos_col in ('ranking', 'position', 'pos'):
                        pv = race_row.get(pos_col, None)
                        if pv is not None and pd.notna(pv):
                            try:
                                run_pos = int(float(pv))
                                break
                            except Exception:
                                pass

                    avg_pm_raw = None
                    avg_pm_str = '—'
                    if race_id is not None and not (isinstance(race_id, float) and pd.isna(race_id)):
                        avg_pm_raw = _rival_avg_pm.get((race_id, hid_raw))
                        if avg_pm_raw is not None:
                            avg_pm_str = (f'€{avg_pm_raw/1000:.1f}k'
                                          if avg_pm_raw >= 1000 else f'€{avg_pm_raw:.0f}')

                    # CHANGE 1: VAL = handicapRatingKg - today_weightKg + 55
                    val_str = '—'
                    val_raw = None
                    try:
                        hcap_v = race_row.get('handicapRatingKg', None)
                        if hcap_v is not None and pd.notna(hcap_v):
                            hcap_float = float(hcap_v)
                            if today_wt is not None:
                                val_raw = hcap_float - today_wt + 55.0
                            else:
                                val_raw = hcap_float
                            val_str = f'{val_raw:.1f}'
                    except Exception:
                        pass

                    run_ranking = None
                    run_runners = None
                    run_lengths = None
                    try:
                        rv = race_row.get('ranking', None)
                        if rv is not None and pd.notna(rv):
                            run_ranking = int(float(rv))
                    except Exception:
                        pass
                    try:
                        rv = race_row.get('runners', None)
                        if rv is not None and pd.notna(rv):
                            run_runners = int(float(rv))
                    except Exception:
                        pass
                    try:
                        rv = race_row.get('cumulative_lengths_back', None)
                        if rv is not None and pd.notna(rv):
                            fv = float(rv)
                            run_lengths = f'{fv:.1f}' if fv != 0 else None
                    except Exception:
                        pass

                    # ARR: adjusted by today's weight (ARR + 55 - today_weightKg)
                    arr_raw = None
                    arr_str = '—'
                    try:
                        arr_v = race_row.get('ARR', None)
                        if arr_v is not None and pd.notna(arr_v):
                            arr_float = float(arr_v)
                            if today_wt is not None:
                                arr_raw = arr_float + (55.0 - today_wt)
                            else:
                                arr_raw = arr_float
                            arr_str = f'{arr_raw:.1f}'
                    except Exception:
                        pass

                    # OPP: today's starters who also ran in this historical race
                    opp_entries = []
                    if race_id is not None and not (isinstance(race_id, float) and pd.isna(race_id)):
                        _race_field = _race_field_cache.get(race_id, pd.DataFrame())
                        for _, _opp_row in _race_field.iterrows():
                            _opp_hid_raw = _opp_row.get('horseId', None)
                            if _opp_hid_raw is None or pd.isna(_opp_hid_raw):
                                continue
                            try:
                                _opp_hid_str = str(int(_opp_hid_raw))
                            except Exception:
                                _opp_hid_str = str(_opp_hid_raw)
                            # Must be a today starter but not the current horse
                            if _opp_hid_str not in today_starter_names:
                                continue
                            if _opp_hid_str == hid_str:
                                continue
                            _opp_name = today_starter_names[_opp_hid_str]
                            # today weights
                            _wt_A_today = horse_today_weight.get(hid_str)        # today weight of THIS horse (A)
                            _wt_B_today = horse_today_weight.get(_opp_hid_str)   # today weight of opponent (B)
                            # historical weights
                            _wt_A_hist = None
                            _wt_B_hist = None
                            try:
                                v = race_row.get('weightKg', None)
                                if v is not None and pd.notna(v):
                                    _wt_A_hist = float(v)
                            except Exception:
                                pass
                            try:
                                v = _opp_row.get('weightKg', None)
                                if v is not None and pd.notna(v):
                                    _wt_B_hist = float(v)
                            except Exception:
                                pass
                            # historical lengths behind winner
                            _lb_A = None
                            _lb_B = None
                            try:
                                v = race_row.get('cumulative_lengths_back', None)
                                if v is not None and pd.notna(v):
                                    _lb_A = float(v)
                            except Exception:
                                pass
                            try:
                                v = _opp_row.get('cumulative_lengths_back', None)
                                if v is not None and pd.notna(v):
                                    _lb_B = float(v)
                            except Exception:
                                pass
                            # Compute adjustment if all values available
                            # Formula: (wt_B_today - wt_A_today) + (lb_B - lb_A) + (wt_A_hist - wt_B_hist)
                            if all(x is not None for x in [_wt_A_today, _wt_B_today, _lb_A, _lb_B, _wt_A_hist, _wt_B_hist]):
                                _adj = (_wt_B_today - _wt_A_today) + (_lb_B - _lb_A) + (_wt_A_hist - _wt_B_hist)
                                opp_entries.append((_opp_name, round(_adj, 1)))

                    ctx_entries.append({
                        'race_id':       race_id,
                        'date_str':      run_date_str,
                        'cond_flag':     cond_flag,
                        'going_cat_raw': str(race_row.get('going_category', '') or ''),
                        'dist_grp_raw':  str(race_row.get('distance_group', '') or ''),
                        'dist_m_raw':    (float(race_row['distance']) if race_row.get('distance') is not None and pd.notna(race_row.get('distance')) else None),
                        'pos':        run_pos,
                        'ranking':    run_ranking,
                        'runners':    run_runners,
                        'lengths':    run_lengths,
                        'avg_pm_str': avg_pm_str,
                        'avg_pm_raw': avg_pm_raw,
                        'val_str':    val_str,
                        'val_raw':    val_raw,
                        'arr_raw':    arr_raw,
                        'arr_str':    arr_str,
                        'info_str':   _build_info_str(race_row),
                        'opp':        opp_entries,
                        'liveOdd':    float(race_row.get('liveOdd')) if race_row.get('liveOdd') is not None and pd.notna(race_row.get('liveOdd')) else None,
                        'jockey_name': str(race_row.get('jockeyName', '') or '').strip() or None,
                        'draw_val':   str(race_row.get('draw_unique', '') or race_row.get('draw', '') or '').strip() or None,
                        'wt_hist':    float(race_row.get('weightKg')) if race_row.get('weightKg') is not None and pd.notna(race_row.get('weightKg')) else None,
                        'lb_raw':     float(race_row.get('cumulative_lengths_back', 0) or 0),
                        'meeting':    (str(race_row.get('name_meeting', '') or race_row.get('meetingName', '') or '').strip() or None),
                        'race_class': (str(race_row.get('class', '') or '').strip() or None),
                        'race_type':  (str(race_row.get('type', '') or '').strip() or None),
                        'opp2':       None,  # filled in post-processing after _race_to_participants is built
                    })
                horse_form_context[hid_str] = ctx_entries

    # ── percentile colour for €/R ──────────────────────────────────────
    all_run_pm_vals = []
    for _hid_str, ctx_list in horse_form_context.items():
        for entry in ctx_list:
            if entry['avg_pm_raw'] is not None:
                all_run_pm_vals.append(entry['avg_pm_raw'])
    all_run_pm_vals_sorted = sorted(all_run_pm_vals)
    _N_pm = len(all_run_pm_vals_sorted)

    def _pm_percentile_color(val):
        if val is None or _N_pm < 2:
            return '#f0f4fa', '#555'
        rank = _bisect.bisect_left(all_run_pm_vals_sorted, val)
        pct  = (rank + 1) / _N_pm
        if pct >= 0.80:   return '#c8f0d8', '#1a6b38'
        elif pct >= 0.60: return '#e2f5ea', '#2d8a50'
        elif pct >= 0.40: return '#f4f7fb', '#4a5568'
        elif pct >= 0.20: return '#fdf0ee', '#b84c3a'
        else:             return '#fde0dc', '#9b2b1a'

    # ── percentile colour for ARR ──────────────────────────────────────
    all_arr_vals = []
    for _hid_str, ctx_list in horse_form_context.items():
        for entry in ctx_list:
            if entry['arr_raw'] is not None:
                all_arr_vals.append(entry['arr_raw'])
    all_arr_vals_sorted = sorted(all_arr_vals)
    _N_arr = len(all_arr_vals_sorted)

    def _arr_percentile_color(val):
        if val is None or _N_arr < 2:
            return '#f0f4fa', '#555'
        rank = _bisect.bisect_left(all_arr_vals_sorted, val)
        pct  = (rank + 1) / _N_arr
        if pct >= 0.80:   return '#c8f0d8', '#1a6b38'
        elif pct >= 0.60: return '#e2f5ea', '#2d8a50'
        elif pct >= 0.40: return '#f4f7fb', '#4a5568'
        elif pct >= 0.20: return '#fdf0ee', '#b84c3a'
        else:             return '#fde0dc', '#9b2b1a'

    # CHANGE 1: percentile colour for VAL (same 5-band logic, across-race normalisation)
    all_val_vals = []
    for _hid_str, ctx_list in horse_form_context.items():
        for entry in ctx_list:
            if entry['val_raw'] is not None:
                all_val_vals.append(entry['val_raw'])
    all_val_vals_sorted = sorted(all_val_vals)
    _N_val = len(all_val_vals_sorted)

    def _val_percentile_color(val):
        if val is None or _N_val < 2:
            return '#f0f4fa', '#555'
        rank = _bisect.bisect_left(all_val_vals_sorted, val)
        pct  = (rank + 1) / _N_val
        if pct >= 0.80:   return '#c8f0d8', '#1a6b38'
        elif pct >= 0.60: return '#e2f5ea', '#2d8a50'
        elif pct >= 0.40: return '#f4f7fb', '#4a5568'
        elif pct >= 0.20: return '#fdf0ee', '#b84c3a'
        else:             return '#fde0dc', '#9b2b1a'

    def _s(r, col):
        v = r.get(col, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '—'
        return str(v)

    def _s_num(r, col, fmt='{:.1f}'):
        v = r.get(col, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '—'
        try:    return fmt.format(float(v))
        except: return str(v)

    def _percentile_circle(val, all_vals):
        if val is None or not all_vals or len(all_vals) < 2:
            return ''
        sorted_vals = sorted(v for v in all_vals if v is not None)
        if len(sorted_vals) < 2 or sorted_vals[-1] == sorted_vals[0]:
            return ''
        pct = (val - sorted_vals[0]) / (sorted_vals[-1] - sorted_vals[0])
        if pct <= 0.5:
            r2, g2, b2 = 220, int(pct * 2 * 200), 0
        else:
            r2, g2, b2 = int((1 - pct) * 2 * 220), 180, 0
        color = f'rgb({r2},{g2},{b2})'
        return (
            f'<svg width="9" height="9" viewBox="0 0 9 9" '
            f'style="vertical-align:middle;margin-left:2px">'
            f'<circle cx="4.5" cy="4.5" r="4" fill="{color}" '
            f'stroke="rgba(0,0,0,.15)" stroke-width="0.5"/></svg>'
        )

    def _hot_cold_html(name, hc_dict):
        if not name or name == '—':
            return ''
        st = hc_dict.get(str(name), None)
        if st is None:
            return ''
        avg_pp = st['avg_pos_perc']
        runs   = st['runs']
        wins   = st['wins']
        places = st['places']
        if avg_pp >= 0.66:   icon = '🔥🔥'
        elif avg_pp >= 0.55: icon = '🔥'
        elif avg_pp <= 0.33: icon = '🧊🧊'
        elif avg_pp <= 0.45: icon = '🧊'
        else: return ''
        tooltip = f'Last 21d: avg pos_perc={avg_pp:.3f} | {runs}R/{wins}W/{places}P'
        return (
            '<span title="' + tooltip + '" style="display:inline-flex;align-items:center;'
            'gap:2px;font-size:11px;cursor:default;margin-left:4px">'
            + icon +
            '<span style="font-size:9px;color:#666;background:rgba(255,255,255,0.85);'
            'border-radius:4px;padding:0 2px;border:1px solid #ddd;white-space:nowrap">'
            + f'{runs}R/{wins}W/{places}P' + '</span></span>'
        )

    def _odds_strip_html(horse_name):
        import re as _re2
        safe_id = _re2.sub(r'[^A-Za-z0-9]', '_', str(horse_name))
        points = horse_odds_strip.get(str(horse_name), [])
        if not points:
            return f'<!--PMU_START:{safe_id}--><!--PMU_END:{safe_id}-->'
        chips = []
        prev_odds = None
        for ts, odds_val in points:
            try:
                odds_f = float(odds_val)
            except (TypeError, ValueError):
                continue
            if prev_odds is None:
                col, arrow = '#888', ''
            elif odds_f < prev_odds:
                col, arrow = '#1a7a3a', '▼'
            elif odds_f > prev_odds:
                col, arrow = '#c0392b', '▲'
            else:
                col, arrow = '#888', ''
            ts_str = ts.strftime('%H:%M') if hasattr(ts, 'strftime') else str(ts)
            arrow_span = f'<span style="font-size:8px">{arrow}</span>' if arrow else ''
            chips.append(
                f'<span title="{ts_str}" style="white-space:nowrap;color:{col};font-size:10px">'
                f'{arrow_span}<strong style="color:{col}">{odds_f:.1f}</strong></span>'
            )
            prev_odds = odds_f
        if not chips:
            return f'<!--PMU_START:{safe_id}--><!--PMU_END:{safe_id}-->'
        joined = '<span style="color:#d0d4dc;font-size:9px;margin:0 2px">›</span>'.join(chips)
        inner = (
            f'<span style="display:inline-flex;align-items:center;gap:2px;'
            f'background:#f8f9fc;border:1px solid #e0e4ec;border-radius:8px;'
            f'padding:1px 7px;margin-left:4px">'
            f'<span style="font-size:9px;color:#aaa;margin-right:3px">PMU</span>'
            f'{joined}</span>'
        )
        return f'<!--PMU_START:{safe_id}-->{inner}<!--PMU_END:{safe_id}-->'

    def _badges_html(st, cat='horse'):
        if not st:
            return '<span style="font-size:10px;color:#bbb;font-style:italic">No qualifying runs</span>'
        runs_v = st.get('runs', 0)
        wins_v = st.get('wins', 0)
        plc_v  = st.get('places', 0)
        ae_v   = st.get('ae_place')
        pm_v   = st.get('prizemoney')

        def _b(label, val, bg='#f0f4fa', col=c_navy, circle=''):
            return (
                '<span style="display:inline-flex;align-items:center;gap:2px;'
                'background:' + bg + ';border-radius:8px;padding:1px 6px;'
                'font-size:10px;border:1px solid ' + c_border + ';white-space:nowrap">'
                '<span style="color:#888">' + label + '</span>'
                '<strong style="color:' + col + '">' + val + '</strong>'
                + circle + '</span>'
            )

        ae_bg  = '#d4edda' if ae_v and ae_v >= 1.0 else '#fde8e8' if ae_v and ae_v < 0.9 else '#f0f4fa'
        ae_col = c_green   if ae_v and ae_v >= 1.0 else '#c0392b' if ae_v and ae_v < 0.9 else c_navy
        ae_str = f'{ae_v:.2f}' if ae_v is not None else '—'
        ae_circle = _percentile_circle(ae_v, cat_ae[cat])
        pm_str    = ('€' + f'{pm_v:,.0f}') if pm_v is not None else '—'
        pm_label  = '€/R(med)' if cat == 'sire' else '€/R'  # CHANGE 5: label sire as median
        pm_circle = _percentile_circle(pm_v, cat_pm[cat])
        w_pct = round(100 * wins_v / runs_v, 1) if runs_v > 0 else 0.0
        p_pct = round(100 * plc_v  / runs_v, 1) if runs_v > 0 else 0.0
        rwp_str = f'{runs_v}/{wins_v}/{plc_v} {w_pct}% {p_pct}%'
        return (
            '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:4px">'
            + _b('R/W/P', rwp_str)
            + _b('A/E', ae_str, ae_bg, ae_col, ae_circle)
            + _b(pm_label, pm_str, '#f0f4fa', c_navy, pm_circle)
            + '</div>'
        )

    def _horse_condition_panel_html(hid_str, today_dist_grp, today_going_grp):
        """Lower panel: R/W/P breakdown by distance_group and going_category."""
        cond = horse_condition_stats.get(hid_str, {})
        if not cond:
            return ''

        def _cond_row(label, breakdown, today_val):
            if not breakdown:
                return ''
            rows_html = ''
            for key, st in sorted(breakdown.items(), key=lambda x: x[0]):
                r = st.get('runs', 0)
                w = st.get('wins', 0)
                p = st.get('places', 0)
                w_pct = round(100 * w / r, 0) if r > 0 else 0
                p_pct = round(100 * p / r, 0) if r > 0 else 0
                is_today = (today_val and str(key).strip() == str(today_val).strip())
                bg     = '#d4edda' if is_today else 'transparent'
                fw     = 'bold'    if is_today else 'normal'
                marker = '▶ '      if is_today else ''
                rows_html += (
                    f'<tr style="background:{bg}">'
                    f'<td style="padding:1px 5px;font-size:9px;color:#555;white-space:nowrap;'
                    f'font-weight:{fw};border-right:1px solid {c_border}">{marker}{key}</td>'
                    f'<td style="padding:1px 5px;font-size:9px;text-align:center;'
                    f'border-right:1px solid {c_border}">{r}</td>'
                    f'<td style="padding:1px 5px;font-size:9px;text-align:center;'
                    f'border-right:1px solid {c_border}">{w}/{p}</td>'
                    f'<td style="padding:1px 5px;font-size:9px;text-align:center;'
                    f'color:{"#1a7a3a" if w_pct >= 20 else "#444"}">{w_pct:.0f}%/{p_pct:.0f}%</td>'
                    f'</tr>'
                )
            if not rows_html:
                return ''
            return (
                f'<div style="margin-top:4px">'
                f'<div style="font-size:9px;color:#999;text-transform:uppercase;'
                f'letter-spacing:.04em;font-weight:bold;margin-bottom:2px">{label}</div>'
                f'<table style="border-collapse:collapse;width:100%;font-family:\'Helvetica Neue\',Arial,sans-serif">'
                f'<thead><tr>'
                f'<th style="padding:1px 5px;font-size:8px;color:#aaa;text-align:left;'
                f'border-right:1px solid {c_border};border-bottom:1px solid {c_border}"></th>'
                f'<th style="padding:1px 5px;font-size:8px;color:#aaa;text-align:center;'
                f'border-right:1px solid {c_border};border-bottom:1px solid {c_border}">R</th>'
                f'<th style="padding:1px 5px;font-size:8px;color:#aaa;text-align:center;'
                f'border-right:1px solid {c_border};border-bottom:1px solid {c_border}">W/P</th>'
                f'<th style="padding:1px 5px;font-size:8px;color:#aaa;text-align:center;'
                f'border-bottom:1px solid {c_border}">W%/P%</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>'
                f'</div>'
            )

        dist_html  = _cond_row('Distanz', cond.get('by_distance', {}), today_dist_grp)
        going_html = _cond_row('Boden',   cond.get('by_going',    {}), today_going_grp)

        if not dist_html and not going_html:
            return ''

        return (
            '<div style="margin-top:6px;padding-top:6px;'
            'border-top:1px dashed ' + c_border + '">'
            + dist_html
            + going_html
            + '</div>'
        )

    def _entity_col(label, name_val, st_dict, cat, prefs_html='', border_right=True, hc_html='', avg_pp=None, all_pp=None):
        border = 'border-right:1px solid ' + c_border + ';' if border_right else ''
        # pp365 badge — always grey background/text, only the percentile circle is coloured
        pp_badge = ''
        if avg_pp is not None:
            pp_circle = _percentile_circle(avg_pp, all_pp or [])
            pp_badge = (
                '<span style="display:inline-flex;align-items:center;gap:2px;'
                'background:#f0f4fa;border-radius:8px;padding:1px 6px;'
                'font-size:10px;border:1px solid ' + c_border + ';white-space:nowrap;margin-top:2px">'
                '<span style="color:#888">pp365</span>'
                '<strong style="color:' + c_navy + '">' + f'{avg_pp:.3f}' + '</strong>'
                + pp_circle + '</span>'
            )

        st = st_dict or {}
        runs_v = st.get('runs', 0)
        wins_v = st.get('wins', 0)
        plc_v  = st.get('places', 0)
        ae_v   = st.get('ae_place')
        pm_v   = st.get('prizemoney')

        def _b(lbl, val, bg='#f0f4fa', col=c_navy, circle=''):
            return (
                '<span style="display:inline-flex;align-items:center;gap:2px;'
                'background:' + bg + ';border-radius:8px;padding:1px 6px;'
                'font-size:10px;border:1px solid ' + c_border + ';white-space:nowrap">'
                '<span style="color:#888">' + lbl + '</span>'
                '<strong style="color:' + col + '">' + val + '</strong>'
                + circle + '</span>'
            )

        if st:
            ae_bg2  = '#d4edda' if ae_v and ae_v >= 1.0 else '#fde8e8' if ae_v and ae_v < 0.9 else '#f0f4fa'
            ae_col2 = c_green   if ae_v and ae_v >= 1.0 else '#c0392b' if ae_v and ae_v < 0.9 else c_navy
            ae_str  = f'{ae_v:.2f}' if ae_v is not None else '—'
            ae_circle = _percentile_circle(ae_v, cat_ae.get(cat, []))
            pm_str   = ('€' + f'{pm_v:,.0f}') if pm_v is not None else '—'
            pm_label = '€/R(med)' if cat == 'sire' else '€/R'
            pm_circle = _percentile_circle(pm_v, cat_pm.get(cat, []))
            w_pct = round(100 * wins_v / runs_v, 1) if runs_v > 0 else 0.0
            p_pct = round(100 * plc_v  / runs_v, 1) if runs_v > 0 else 0.0
            rwp_str = f'{runs_v}/{wins_v}/{plc_v} {w_pct}% {p_pct}%'
            badges = (
                '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:4px">'
                + _b('R/W/P', rwp_str)
                + _b('A/E', ae_str, ae_bg2, ae_col2, ae_circle)
                + _b(pm_label, pm_str, '#f0f4fa', c_navy, pm_circle)
                + ('  ' + pp_badge if pp_badge else '')
                + '</div>'
            )
        else:
            badges = (
                '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-top:4px">'
                '<span style="font-size:10px;color:#bbb;font-style:italic">No qualifying runs</span>'
                + ('  ' + pp_badge if pp_badge else '')
                + '</div>'
            )

        return (
            '<div style="flex:1;min-width:0;padding:0 12px;' + border +
            'display:flex;flex-direction:column;gap:2px">'
            '<div style="font-size:10px;color:#999;text-transform:uppercase;'
            'letter-spacing:.05em;font-weight:bold">' + label + '</div>'
            '<div style="font-weight:bold;color:' + c_navy + ';font-size:12px;'
            'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            'display:flex;align-items:center;gap:2px" title="' + name_val + '">'
            + name_val + hc_html + '</div>'
            + badges
            + prefs_html + '</div>'
        )

    def _form_context_table_html(hid_str):
            ctx_list = horse_form_context.get(hid_str, [])
            if not ctx_list:
                return ''

            MEDAL_BG = {1: '#C9960C', 2: '#888888', 3: '#A0522D'}

            col_w = '52px'

            date_cells = ''.join(
                f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                f'color:#666;white-space:nowrap;border-right:1px solid {c_border};'
                f'min-width:{col_w}">{e["date_str"]}</td>'
                for e in ctx_list
            )

            flag_cells = ''
            for e in ctx_list:
                flag = e['cond_flag']
                pos  = e['pos']
                medal_bg  = MEDAL_BG.get(pos, 'transparent') if pos else 'transparent'
                medal_txt = 'white' if pos in (1, 2, 3) else ('#444' if flag else '#bbb')
                if not flag:
                    flag      = '·'
                    medal_txt = '#ccc'
                flag_cells += (
                    f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                    f'font-weight:bold;border-right:1px solid {c_border};min-width:{col_w}">'
                    f'<span style="background:{medal_bg};color:{medal_txt};'
                    f'border-radius:50%;display:inline-block;'
                    f'min-width:24px;line-height:20px;text-align:center;'
                    f'padding:0 3px;font-size:9px">{flag}</span></td>'
                )

            result_cells = ''
            for e in ctx_list:
                rnk  = e.get('ranking')
                rnrs = e.get('runners')
                lngs = e.get('lengths')
                pos  = e.get('pos')
                lo   = e.get('liveOdd')

                race_id_str = str(e.get('race_id', '')) if e.get('race_id') is not None else ''
                is_flagged = (
                    notepad_flags is not None
                    and notepad_flags.get((race_id_str, hid_str), False)
                )

                res_str   = (str(rnk) + (f'/{rnrs}' if rnrs is not None else '')) if rnk is not None else '—'
                lngs_html = (f'<br><span style="color:#999;font-size:9px">({lngs}l)</span>' if lngs else '')
                lo_html   = (f'<br><span style="color:#555;font-size:9px">{lo:.1f}/1</span>' if lo is not None else '')
                pos_bg  = MEDAL_BG.get(pos, 'transparent') if pos in (1, 2, 3) else 'transparent'
                pos_col = 'white' if pos in (1, 2, 3) else '#444'
                pos_br  = '3px'   if pos in (1, 2, 3) else '0'
                notepad_html = (
                    '<span style="position:absolute;top:1px;right:2px;font-size:9px;'
                    'line-height:1;opacity:0.85" title="Notepad flag: unlucky run or strong finish">'
                    '📓</span>'
                    if is_flagged else ''
                )
                result_cells += (
                    f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                    f'border-right:1px solid {c_border};min-width:{col_w};line-height:1.3;'
                    f'position:relative">'
                    f'<span style="background:{pos_bg};color:{pos_col};'
                    f'border-radius:{pos_br};padding:1px 3px;font-weight:bold;font-size:10px">'
                    f'{res_str}</span>{lngs_html}{lo_html}{notepad_html}</td>')

            # Jcky row: jockey name only (abbreviated), no pp
            jcky_cells = ''
            for e in ctx_list:
                jname = e.get('jockey_name') or ''
                if jname:
                    parts = jname.split()
                    abbr  = (parts[0][0] + '. ' + parts[-1]) if len(parts) >= 2 else jname
                    if len(abbr) > 14:
                        abbr = abbr[:13] + '…'
                    jcky_cells += (
                        f'<td style="padding:3px 5px;text-align:center;'
                        f'border-right:1px solid {c_border};min-width:{col_w};line-height:1.4">'
                        f'<span style="font-size:9px;white-space:nowrap">{abbr}</span></td>'
                    )
                else:
                    jcky_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                        f'color:#ccc;border-right:1px solid {c_border};min-width:{col_w}">—</td>'
                    )

            # Draw row: draw number only (strip everything after " - ") + draw bias
            draw_cells = ''
            for e in ctx_list:
                dv_raw = e.get('draw_val') or ''
                # Extract only the leading number before any " - "
                dv = dv_raw.split(' - ')[0].strip() if dv_raw else ''
                if dv:
                    dp = draw_pos_perc.get(str(dv_raw))  # lookup uses the full key
                    if dp is not None:
                        dp_val, dp_n = dp
                        sign = '+' if dp_val >= 0 else ''
                        dp_col2 = '#1a7a3a' if dp_val > 0 else '#c0392b' if dp_val < 0 else '#888'
                        inner = (
                            f'<span style="font-size:10px">{dv}</span>'
                            f'<br><strong style="font-size:9px;color:{dp_col2}">{sign}{dp_val:.3f}</strong>'
                        )
                    else:
                        inner = f'<span style="font-size:10px">{dv}</span>'
                    draw_cells += (
                        f'<td style="padding:3px 5px;text-align:center;'
                        f'border-right:1px solid {c_border};min-width:{col_w};line-height:1.4">'
                        f'{inner}</td>'
                    )
                else:
                    draw_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                        f'color:#ccc;border-right:1px solid {c_border};min-width:{col_w}">—</td>'
                    )

            pm_cells = ''.join(
                f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                f'font-weight:600;background:{_pm_percentile_color(e["avg_pm_raw"])[0]};'
                f'color:{_pm_percentile_color(e["avg_pm_raw"])[1]};'
                f'border-right:1px solid {c_border};min-width:{col_w};white-space:nowrap">'
                f'{e["avg_pm_str"]}</td>'
                for e in ctx_list
            )

            val_cells = ''.join(
                f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                f'font-weight:600;background:{_val_percentile_color(e["val_raw"])[0]};'
                f'color:{_val_percentile_color(e["val_raw"])[1]};'
                f'border-right:1px solid {c_border};min-width:{col_w};white-space:nowrap">'
                f'{e["val_str"]}</td>'
                for e in ctx_list
            )

            arr_cells = ''.join(
                f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                f'font-weight:600;background:{_arr_percentile_color(e["arr_raw"])[0]};'
                f'color:{_arr_percentile_color(e["arr_raw"])[1]};'
                f'border-right:1px solid {c_border};min-width:{col_w};white-space:nowrap">'
                f'{e["arr_str"]}</td>'
                for e in ctx_list
            )

            # FF row: only for first 3 races, rest get '—'
            ff_cells = ''
            for i, e in enumerate(ctx_list):
                race_id = e.get('race_id', None)
                if i < 3 and race_id is not None:
                    ff = ff_stats_by_race.get(race_id, None)
                    if ff is not None:
                        avg_d = ff['avg_delta']
                        n_ff  = ff['n']
                        ff_col  = '#1a7a3a' if avg_d >= 0 else '#c0392b'
                        ff_bg   = '#d4edda' if avg_d >= 0 else '#fde8e8'
                        sign    = '+' if avg_d >= 0 else ''
                        ff_str  = f'{sign}{avg_d:.2f}'
                        ff_cells += (
                            f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                            f'font-weight:600;background:{ff_bg};color:{ff_col};'
                            f'border-right:1px solid {c_border};min-width:{col_w};white-space:nowrap" '
                            f'title="Avg ARR delta of field members who ran again (n={n_ff})">'
                            f'{ff_str}'
                            f'<span style="font-size:9px;color:#999;margin-left:2px">n={n_ff}</span>'
                            f'</td>'
                        )
                    else:
                        ff_cells += (
                            f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                            f'color:#ccc;border-right:1px solid {c_border};min-width:{col_w}">—</td>'
                        )
                else:
                    ff_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                        f'color:#ccc;border-right:1px solid {c_border};min-width:{col_w}">—</td>'
                    )

            # Opp row: today's starters who ran in the same historical race
            opp_cells = ''
            for e in ctx_list:
                opp_list = e.get('opp', [])
                if opp_list:
                    parts = []
                    for opp_name, adj in sorted(opp_list, key=lambda x: abs(x[1]), reverse=True):
                        sign = '+' if adj >= 0 else ''
                        col  = '#1a7a3a' if adj >= 0 else '#c0392b'
                        # Shorten name: first word only
                        short = opp_name.split()[0] if opp_name else opp_name
                        parts.append(
                            f'<span style="white-space:nowrap;font-size:9px">'
                            f'{short} <strong style="color:{col}">{sign}{adj:.1f}</strong></span>'
                        )
                    cell_inner = '<br>'.join(parts)
                    opp_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:9px;'
                        f'color:#444;border-right:1px solid {c_border};min-width:{col_w};'
                        f'line-height:1.4" title="Gegner aus heutigem Starterfeld (Adj.)">'
                        f'{cell_inner}</td>'
                    )
                else:
                    opp_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                        f'color:#ccc;border-right:1px solid {c_border};min-width:{col_w}">—</td>'
                    )

            # Opp2 row: up to 3 indirect links per race column, sorted by date desc
            opp2_cells = ''
            for e in ctx_list:
                o2_list = e.get('opp2') or []
                if o2_list:
                    parts = []
                    for o2 in o2_list:
                        score  = o2['score']
                        b_name = o2['b_name']
                        x_name = o2['x_name']
                        date_x = o2['date_x']
                        b_short = b_name.split()[0] if b_name else b_name
                        x_short = x_name.split()[0] if x_name else x_name
                        sign   = '+' if score >= 0 else ''
                        s_col  = '#1a7a3a' if score >= 0 else '#c0392b'
                        parts.append(
                            f'<span style="white-space:nowrap;font-size:9px;color:#555">'
                            f'{b_short}'
                            f'<span style="color:#aaa;font-size:8px"> v {x_short}</span> '
                            f'<strong style="color:{s_col}">{sign}{score:.1f}</strong>'
                            f'<span style="color:#aaa;font-size:8px"> ({date_x})</span>'
                            f'</span>'
                        )
                    # background based on first (most recent) entry
                    opp2_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:9px;'
                        f'border-right:1px solid {c_border};'
                        f'min-width:{col_w};line-height:1.5">'
                        + '<br>'.join(parts)
                        + '</td>'
                    )
                else:
                    opp2_cells += (
                        f'<td style="padding:3px 5px;text-align:center;font-size:10px;'
                        f'color:#ccc;border-right:1px solid {c_border};min-width:{col_w}">—</td>'
                    )
            label_style = (
                f'padding:3px 6px;text-align:right;font-size:9px;'
                f'color:#999;font-weight:bold;text-transform:uppercase;'
                f'letter-spacing:.04em;white-space:nowrap;'
                f'border-right:1px solid {c_border};background:#fafbfd'
            )

            info_cells = ''.join(
                f'<td style="padding:3px 5px;text-align:center;font-size:9px;'
                f'color:#555;border-right:1px solid {c_border};'
                f'min-width:{col_w};line-height:1.3"'
                f' title="{e["info_str"].replace("<br>", " | ")}">{e["info_str"]}</td>'
                for e in ctx_list
            )

            return (
                f'<div style="overflow-x:auto">'
                f'<table style="border-collapse:collapse;font-family:\'Helvetica Neue\',Arial,sans-serif;'
                f'border:1px solid {c_border};border-radius:4px;font-size:10px;width:auto">'
                f'<tbody>'
                f'<tr><td style="{label_style}">Date</td>{date_cells}</tr>'
                f'<tr style="background:#f8f9fc"><td style="{label_style}">Info</td>{info_cells}</tr>'
                f'<tr><td style="{label_style}">Cond</td>{flag_cells}</tr>'
                f'<tr style="background:#f8f9fc"><td style="{label_style}">€/R</td>{pm_cells}</tr>'
                f'<tr><td style="{label_style}">Res</td>{result_cells}</tr>'
                f'<tr style="background:#f8f9fc"><td style="{label_style}">Val.</td>{val_cells}</tr>'
                f'<tr><td style="{label_style}">ARR</td>{arr_cells}</tr>'
                f'<tr style="background:#f8f9fc"><td style="{label_style}">FF</td>{ff_cells}</tr>'
                f'<tr><td style="{label_style}">Jcky</td>{jcky_cells}</tr>'
                f'<tr style="background:#f8f9fc"><td style="{label_style}">Draw</td>{draw_cells}</tr>'
                f'<tr><td style="{label_style}">Opp</td>{opp_cells}</tr>'
                f'<tr style="background:#f8f9fc"><td style="{label_style}">Opp2</td>{opp2_cells}</tr>'
                f'</tbody>'
                f'</table>'
                f'</div>'
            )

    # ── Opp2: per-race best indirect link (A via X vs B) ───────────────────────
    # For each historical race of horse A, find the best link to any today-starter B
    # via a horse X that ran in that same race. Both races must be within 365 days
    # and no horse may be tailed off (lb > 8). One result per form-table column.
    # Result stored in ctx_entry['opp2'] = {'b_name', 'x_name', 'score', 'date_x'}

    if runners_hist is not None and 'horseId' in rows.columns and horse_form_context:
        try:
            _TO_OPP2   = 8.0
            _cutoff_ts = pd.Timestamp(cutoff_365)
            _today_hids_set = set(str(int(h)) for h in rows['horseId'].dropna().unique())

            # Build participation index: race_id -> {hid_str: (wt, lb, date)}
            _be_cols  = ['horseId', 'raceId', 'weightKg', 'cumulative_lengths_back', '_fdt']
            _be_avail = [c for c in _be_cols if c in _all_hist.columns]
            _rtp = {}   # race_id -> {hid_str: (wt, lb, date)}
            _hid_to_name = {}  # hid_str -> horseName

            if len(_be_avail) >= 3:
                _be_df2 = _all_hist[_be_avail].dropna(subset=['horseId', 'raceId']).copy()
                for _, _br in _be_df2.iterrows():
                    _bh = str(int(_br['horseId'])) if pd.notna(_br['horseId']) else None
                    _brid = _br['raceId']
                    if not _bh or _brid is None:
                        continue
                    _bwt = float(_br['weightKg']) if pd.notna(_br.get('weightKg')) else None
                    _blb = float(_br['cumulative_lengths_back']) if pd.notna(_br.get('cumulative_lengths_back')) else 0.0
                    _bdt = _br['_fdt'] if '_fdt' in _br else None
                    if _brid not in _rtp:
                        _rtp[_brid] = {}
                    _rtp[_brid][_bh] = (_bwt, _blb, _bdt)

            # Build horseName lookup from _all_hist
            if 'horseName' in _all_hist.columns and 'horseId' in _all_hist.columns:
                for _, _nr in _all_hist[['horseId','horseName']].dropna().iterrows():
                    try:
                        _hid_to_name[str(int(_nr['horseId']))] = str(_nr['horseName'])
                    except Exception:
                        pass

            # Build per-horse race list for X lookup: hid -> [(rid, wt, lb, date)]
            _h_races = {}
            for _rid2, _pmap in _rtp.items():
                for _bh2, (_bwt2, _blb2, _bdt2) in _pmap.items():
                    if _bh2 not in _h_races:
                        _h_races[_bh2] = []
                    _h_races[_bh2].append((_rid2, _bwt2, _blb2, _bdt2))

            # Post-process each ctx_entry: compute opp2
            for _hid_A, _ctx_list in horse_form_context.items():
                _wt_A_today = horse_today_weight.get(_hid_A)
                if _wt_A_today is None:
                    continue
                for _entry in _ctx_list:
                    _rid_A  = _entry.get('race_id')
                    _wt_A_h = _entry.get('wt_hist')
                    _lb_A   = _entry.get('lb_raw', 0.0)
                    if _rid_A is None or _wt_A_h is None:
                        continue
                    # A must not be tailed off in this race
                    if _lb_A > _TO_OPP2:
                        continue

                    # Each candidate: (dt_xb_raw, score, b_name, x_name, date_xb_fmt,
                    #                  sp_b, lb_diff_ax, hid_b)
                    # lb_diff_ax = abs(lb_A - lb_X_A): how close X was to A in their shared race
                    _candidates = []
                    _seen_bx = set()  # deduplicate (b_name, x_name) pairs
                    _participants_A = _rtp.get(_rid_A, {})

                    for _hid_X, (_wt_X_A, _lb_X_A, _dt_X_A) in _participants_A.items():
                        if _hid_X == _hid_A:
                            continue
                        if _hid_X in _today_hids_set:
                            continue
                        if _wt_X_A is None:
                            continue
                        # X must not be tailed off against A
                        if _lb_X_A > _TO_OPP2:
                            continue
                        # Race of A must be within 365 days
                        if _dt_X_A is None or pd.Timestamp(_dt_X_A) < _cutoff_ts:
                            continue

                        _lb_diff_ax = abs(_lb_A - _lb_X_A)  # closeness of X to A in their race

                        # For each today's starter B, find the most recent race where X ran vs B
                        # Exclude B if B was also in race _rid_A (those go in Opp, not Opp2)
                        _direct_opps = set(_rtp.get(_rid_A, {}).keys()) & _today_hids_set
                        _x_races = _h_races.get(_hid_X, [])
                        for _hid_B in list(_today_hids_set):
                            if _hid_B == _hid_A:
                                continue
                            if _hid_B in _direct_opps:
                                continue  # direct opponent — skip, already in Opp row
                            _wt_B_today = horse_today_weight.get(_hid_B)
                            if _wt_B_today is None:
                                continue
                            for (_rid_XB, _wt_X_B, _lb_X_B, _dt_XB) in _x_races:
                                if _rid_XB == _rid_A:
                                    continue
                                if _dt_XB is None or pd.Timestamp(_dt_XB) < _cutoff_ts:
                                    continue
                                _bp = _rtp.get(_rid_XB, {}).get(_hid_B)
                                if _bp is None:
                                    continue
                                _wt_B_h, _lb_B, _ = _bp
                                if _wt_B_h is None or _wt_X_B is None:
                                    continue
                                if _lb_B > _TO_OPP2 or _lb_X_B > _TO_OPP2:
                                    continue
                                _score_A_X = (_wt_X_A - _wt_A_h) + (_lb_A   - _lb_X_A)
                                _score_B_X = (_wt_X_B - _wt_B_h) + (_lb_B   - _lb_X_B)
                                _raw_diff  = _score_B_X - _score_A_X
                                _today_adj = _wt_B_today - _wt_A_today
                                _final     = round(_raw_diff + _today_adj, 1)
                                _b_name    = today_starter_names.get(_hid_B, _hid_B)
                                _x_name    = _hid_to_name.get(_hid_X, _hid_X)
                                _bx_key    = (_b_name, _x_name)
                                _date_xb_fmt = pd.Timestamp(_dt_XB).strftime('%b %y') if _dt_XB is not None else '?'
                                _sp_b      = today_starter_sp.get(_hid_B, float('inf'))
                                if _bx_key not in _seen_bx:
                                    _seen_bx.add(_bx_key)
                                    _candidates.append((
                                        _dt_XB, _final, _b_name, _x_name, _date_xb_fmt,
                                        _sp_b, _lb_diff_ax, _hid_B
                                    ))
                                break  # one X-B race per (X,B) combo

                    if _candidates:
                        if len(_candidates) > 3:
                            # ── Step 1: sort by SP of horse B descending (best ML win
                            #   chance first), then within same B keep only the X that
                            #   was closest to A (lowest abs length difference A–X).
                            # ── Step 2: one horse B only — from all (X→B) paths choose
                            #   the single X with the smallest abs(lb_A − lb_X_A).

                            # Group all paths by horse B; for each B retain the X that
                            # minimises abs(lb_A − lb_X_A) (i.e. closest X to A).
                            _best_per_b = {}  # hid_B -> best candidate tuple
                            for _c in _candidates:
                                (_c_dt, _c_sc, _c_bn, _c_xn, _c_dx,
                                 _c_sp, _c_lbd, _c_hidb) = _c
                                if _c_hidb not in _best_per_b:
                                    _best_per_b[_c_hidb] = _c
                                else:
                                    _prev = _best_per_b[_c_hidb]
                                    if _c_lbd < _prev[6]:  # lower lb_diff wins
                                        _best_per_b[_c_hidb] = _c

                            # Collect one-per-B candidates, sort by SP descending
                            _deduped = sorted(
                                _best_per_b.values(),
                                key=lambda c: c[5] if c[5] != float('inf') else -1,
                                reverse=True  # highest SP (best ML chance) first
                            )
                            _candidates = _deduped
                        # Take the 3 best
                        _entry['opp2'] = [
                            {'score': sc, 'b_name': bn, 'x_name': xn, 'date_x': dx}
                            for (_, sc, bn, xn, dx, _sp, _lbd, _hidb) in _candidates[:3]
                        ]
        except Exception:
            pass  # silently skip if data insufficient

    import re as _re_safe

    # Pre-build horse→history dict so headgear/trainer alert lookups are O(1)
    _runners_hist_by_horse = {}
    if runners_hist is not None and 'horseId' in runners_hist.columns:
        for _rh_hid, _rh_grp in runners_hist.groupby('horseId'):
            try:
                _rh_key = str(int(_rh_hid))
            except (ValueError, TypeError):
                _rh_key = str(_rh_hid)
            _rh_grp = _rh_grp.copy()
            if 'date' in _rh_grp.columns:
                _rh_grp['_hgdt'] = pd.to_datetime(_rh_grp['date'], errors='coerce')
                _rh_grp = _rh_grp.sort_values('_hgdt', ascending=False)
            _runners_hist_by_horse[_rh_key] = _rh_grp

    _race_json_horses = []
    cards_html = []
    for _, r in rows.iterrows():
        horse   = _s(r, 'horseName')
        trainer = _s(r, 'trainerName')
        jockey  = _s(r, 'jockeyName')
        sire    = _s(r, 'horseSir')
        age     = _s(r, 'age')
        sex     = _s(r, 'sex')
        weight  = _s_num(r, 'weightKg', '{:.1f}')
        draw    = _s(r, 'draw')
        draw_unique_val = r.get('draw_unique', None)
        if draw_unique_val is None or (isinstance(draw_unique_val, float) and pd.isna(draw_unique_val)):
            draw_unique_val = draw
        blink  = _s(r, 'blinkers')
        hood_v = _s(r, 'hood')
        sp     = r.get('SP', None)

        hcap_raw = r.get('handicapRatingKg', None)
        wt_raw   = r.get('weightKg', None)
        adj_val_float = None
        adj_val_str   = '—'
        try:
            if hcap_raw is not None and wt_raw is not None and pd.notna(hcap_raw) and pd.notna(wt_raw):
                adj_val_float = float(hcap_raw) - float(wt_raw) + 55
                adj_val_str   = f'{adj_val_float:.1f}'
        except Exception:
            pass

        # CHANGE 3: rating_after_race adjusted by today's weight — shown in name line
        rtr_val   = r.get('rating_after_race', None)
        rtr_float = None
        rtr_adj_float = None
        rtr_adj_str   = '—'
        if rtr_val is not None and pd.notna(rtr_val):
            try:
                rtr_float = float(rtr_val)
                if wt_raw is not None and pd.notna(wt_raw):
                    rtr_adj_float = rtr_float - float(wt_raw) + 55
                    rtr_adj_str   = f'{rtr_adj_float:.1f}'
                else:
                    rtr_adj_float = rtr_float
                    rtr_adj_str   = f'{rtr_float:.0f}'
            except Exception:
                pass

        hid = (str(int(r['horseId'])) if 'horseId' in r.index and pd.notna(r.get('horseId')) else '')

        h_st = stats_by_horse.get(hid, {})
        t_st = stats_by_trainer.get(trainer if trainer != '—' else '', {})
        j_st = stats_by_jockey.get(jockey   if jockey  != '—' else '', {})
        s_st = stats_by_sire.get(sire        if sire   != '—' else '', {})

        def _row_val(col):
            v = r.get(col, None)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return str(v)
            return None

        horse_dist_grp  = _row_val('distance_group') or today_distance_grp
        horse_going_grp = _row_val('going_category') or today_going_grp
        horse_meeting   = _row_val('name_meeting')   or today_name_meeting
        horse_racetype  = _row_val('type') or _row_val('raceType') or today_racetype

        today_trainer_for_jockey = trainer if trainer != '—' else None
        today_jockey_for_trainer = jockey  if jockey  != '—' else None
        today_owner_for_trainer  = _row_val('ownerName') or None
        horse_age = _row_val('age') or None

        # Compute t-tests once; render HTML from the same results (no duplicate scipy calls)
        _jockey_prefs_data = _build_prefs_data(
            'jockeyName', jockey if jockey != '—' else '', _hist_750,
            [('🎩', 'Trainer', 'trainerName', today_trainer_for_jockey),
             ('📍', 'Meeting', 'name_meeting', horse_meeting)])
        _trainer_prefs_data = _build_prefs_data(
            'trainerName', trainer if trainer != '—' else '', _hist_750,
            [('🏇', 'Jockey',   'jockeyName',  today_jockey_for_trainer),
             ('👤', 'Owner',    'ownerName',   today_owner_for_trainer),
             ('📍', 'Meeting',  'name_meeting', horse_meeting),
             ('🏆', 'RaceType', 'type',         horse_racetype)])
        if not _trainer_prefs_data:
            _trainer_prefs_data = _build_prefs_data(
                'trainerName', trainer if trainer != '—' else '', _hist_750,
                [('🏇', 'Jockey',   'jockeyName',  today_jockey_for_trainer),
                 ('👤', 'Owner',    'ownerName',   today_owner_for_trainer),
                 ('📍', 'Meeting',  'name_meeting', horse_meeting),
                 ('🏆', 'RaceType', 'raceType',    horse_racetype)])
        _horse_prefs_data = _build_prefs_data('horseId', hid, _hist_all,
            [('📍', 'Meeting',  'name_meeting',   horse_meeting),
             ('📏', 'Distance', 'distance_group', horse_dist_grp),
             ('🌱', 'Going',    'going_category', horse_going_grp)])
        _sire_prefs_data = _build_prefs_data(
            'horseSir', sire if sire != '—' else '', _hist_all,
            [('📏', 'Distance', 'distance_group', horse_dist_grp),
             ('🌱', 'Going',    'going_category', horse_going_grp),
             ('🎂', 'Age',      'age',            horse_age)])

        jockey_prefs_html  = _prefs_data_to_html(_jockey_prefs_data)
        trainer_prefs_html = _prefs_data_to_html(_trainer_prefs_data)
        horse_prefs_html   = _prefs_data_to_html(_horse_prefs_data)
        sire_prefs_html    = _prefs_data_to_html(_sire_prefs_data)

        trainer_hc_html = _hot_cold_html(trainer if trainer != '—' else '', hot_cold_trainer)
        jockey_hc_html  = _hot_cold_html(jockey  if jockey  != '—' else '', hot_cold_jockey)

        sp_html = ''
        if sp is not None and pd.notna(sp):
            sp_html = (
                '<span style="background:' + c_navy + ';color:white;'
                'font-size:12px;font-weight:bold;padding:2px 10px;border-radius:12px;'
                'white-space:nowrap">SP ' + f'{float(sp):.1f}' + '</span>'
            )

        equip_parts = []
        if blink != '—' and blink:
            equip_parts.append(
                '<span style="background:#fff3cd;color:#856404;border-radius:8px;'
                'padding:1px 7px;font-size:10px;border:1px solid #ffe08a">B: ' + blink + '</span>')
        if hood_v != '—' and hood_v:
            equip_parts.append(
                '<span style="background:#e8f4fd;color:#1a6fa8;border-radius:8px;'
                'padding:1px 7px;font-size:10px;border:1px solid #b8d9f2">H: ' + hood_v + '</span>')
        equip_html = ' '.join(equip_parts)

        _horse_change_alerts = []   # structured data collected alongside HTML alerts
        headgear_change_html = ''
        _h_hist = _runners_hist_by_horse.get(hid, pd.DataFrame())
        if not _h_hist.empty:
                _last = _h_hist.iloc[0]
                _prev_blink = str(_last.get('blinkers', '')).strip() if pd.notna(_last.get('blinkers')) else ''
                _prev_hood  = str(_last.get('hood', '')).strip()     if pd.notna(_last.get('hood'))     else ''
                _prev_blink = '' if _prev_blink in ('nan','—') else _prev_blink
                _prev_hood  = '' if _prev_hood  in ('nan','—') else _prev_hood
                _today_blink = blink  if blink  not in ('—','') else ''
                _today_hood  = hood_v if hood_v not in ('—','') else ''
                _hg_alerts = []
                if _today_blink != _prev_blink:
                    _label = f'Blinkers: {_prev_blink or "none"} → {_today_blink or "none"}'
                    _hg_alerts.append(
                        '<span style="background:#ff6b35;color:white;border-radius:6px;'
                        'padding:2px 8px;font-size:10px;font-weight:bold;white-space:nowrap;'
                        f'border:1px solid #e55a26" title="{_label}">👓 B-Change</span>'
                    )
                    _horse_change_alerts.append({'type': 'blinkers_change',
                                                 'from': _prev_blink or 'none',
                                                 'to':   _today_blink or 'none'})
                if _today_hood != _prev_hood:
                    _label = f'Hood: {_prev_hood or "none"} → {_today_hood or "none"}'
                    _hg_alerts.append(
                        '<span style="background:#9b59b6;color:white;border-radius:6px;'
                        'padding:2px 8px;font-size:10px;font-weight:bold;white-space:nowrap;'
                        f'border:1px solid #8e44ad" title="{_label}">🎩 H-Change</span>'
                    )
                    _horse_change_alerts.append({'type': 'hood_change',
                                                 'from': _prev_hood or 'none',
                                                 'to':   _today_hood or 'none'})
                headgear_change_html = ' '.join(_hg_alerts)

        trainer_change_html = ''
        owner_change_html   = ''
        _t_hist = _h_hist  # same pre-built sorted history (sorted by _hgdt == _tdt)
        if not _t_hist.empty:
                # ── Trainer change ────────────────────────────────────────────
                if 'trainerName' in _t_hist.columns:
                    _prev_trainer = str(_t_hist.iloc[0].get('trainerName', '')).strip()
                    _prev_trainer = '' if _prev_trainer in ('nan','—') else _prev_trainer
                    _today_trainer = trainer if trainer not in ('—','') else ''
                    if _today_trainer and _prev_trainer and _today_trainer != _prev_trainer:
                        _old_pp   = avg_pp_365_trainer.get(_prev_trainer)
                        _pp_str   = f' {_old_pp:.3f}' if _old_pp is not None else ''
                        _pp_col   = '#1a7a3a' if (_old_pp and _old_pp >= 0.55) else '#c0392b' if (_old_pp and _old_pp <= 0.45) else 'white'
                        _label    = f'Trainer: {_prev_trainer} → {_today_trainer}'
                        trainer_change_html = (
                            '<span style="background:#e74c3c;color:white;border-radius:6px;'
                            'padding:2px 8px;font-size:10px;font-weight:bold;white-space:nowrap;'
                            f'border:1px solid #c0392b;display:inline-flex;align-items:center;gap:4px" title="{_label}">'
                            '🔄 T-Change'
                            f'<span style="font-size:9px;opacity:0.9">{_prev_trainer[:14]}</span>'
                            + (f'<span style="color:{_pp_col};font-size:9px">{_pp_str}</span>' if _pp_str else '')
                            + '</span>'
                        )
                        _horse_change_alerts.append({'type': 'trainer_change',
                                                     'from': _prev_trainer, 'to': _today_trainer,
                                                     'prev_trainer_pp365': _old_pp})

                # ── Owner change ──────────────────────────────────────────────
                if 'ownerName' in _t_hist.columns:
                    _prev_owner  = str(_t_hist.iloc[0].get('ownerName', '')).strip()
                    _prev_owner  = '' if _prev_owner in ('nan','—') else _prev_owner
                    _today_owner = _row_val('ownerName') or ''
                    if _today_owner and _prev_owner and _today_owner != _prev_owner:
                        _old_opp   = avg_pp_365_owner.get(_prev_owner)
                        _opp_str   = f' {_old_opp:.3f}' if _old_opp is not None else ''
                        _opp_col   = '#1a7a3a' if (_old_opp and _old_opp >= 0.55) else '#c0392b' if (_old_opp and _old_opp <= 0.45) else 'white'
                        _olabel    = f'Owner: {_prev_owner} → {_today_owner}'
                        owner_change_html = (
                            '<span style="background:#8e44ad;color:white;border-radius:6px;'
                            'padding:2px 8px;font-size:10px;font-weight:bold;white-space:nowrap;'
                            f'border:1px solid #7d3c98;display:inline-flex;align-items:center;gap:4px" title="{_olabel}">'
                            '👤 O-Change'
                            f'<span style="font-size:9px;opacity:0.9">{_prev_owner[:14]}</span>'
                            + (f'<span style="color:{_opp_col};font-size:9px">{_opp_str}</span>' if _opp_str else '')
                            + '</span>'
                        )
                        _horse_change_alerts.append({'type': 'owner_change',
                                                     'from': _prev_owner, 'to': _today_owner,
                                                     'prev_owner_pp365': _old_opp})

        days_since = horse_last_start.get(hid, None)
        daysince_html = ''
        if days_since is not None:
            long_break = days_since > 60
            ds_bg  = '#fff3cd' if long_break else '#f0f4fa'
            ds_col = '#856404' if long_break else '#666'
            ds_bdr = '#ffe08a' if long_break else c_border
            daysince_html = (
                '<span style="font-size:11px;display:inline-flex;align-items:center;gap:2px;'
                'background:' + ds_bg + ';border-radius:6px;padding:1px 6px;'
                'border:1px solid ' + ds_bdr + ';color:' + ds_col + ';white-space:nowrap" '
                'title="Days since last start">'
                + ('⏸ ' if long_break else '') + f'{days_since}d</span>'
            )

        draw_pp_html = ''
        if draw_unique_val is not None and str(draw_unique_val) != '—':
            dp = draw_pos_perc.get(str(draw_unique_val), None)
            if dp is not None:
                dp_val, dp_n = dp
                dp_col  = '#1a7a3a' if dp_val > 0 else '#c0392b'
                dp_bg   = '#d4edda' if dp_val > 0 else '#fde8e8'
                dp_sign = '+' if dp_val > 0 else ''
                draw_pp_html = (
                    '<span style="font-size:11px;display:inline-flex;align-items:center;gap:2px;'
                    'background:' + dp_bg + ';border-radius:6px;padding:1px 6px;'
                    'border:1px solid #ddd;color:' + dp_col + ';white-space:nowrap" '
                    'title="Draw pos_perc bias (avg - 0.5) | n=' + str(dp_n) + '">'
                    + dp_sign + f'{dp_val:.3f}'
                    + '<span style="font-size:9px;color:#999;margin-left:2px">n=' + str(dp_n) + '</span></span>'
                )

        meta_parts = []
        for lbl, val in [('Age', age), ('Sex', sex)]:
            if val and val != '—':
                meta_parts.append(
                    '<span style="font-size:11px;color:#666">'
                    + lbl + ': <strong>' + val + '</strong></span>'
                )
        if weight != '—':
            meta_parts.append(
                '<span style="font-size:11px;color:#666">Wt: <strong>' + weight + 'kg</strong></span>'
            )
        if adj_val_str != '—':
            val_circle = _percentile_circle(adj_val_float, cat_val)
            meta_parts.append(
                '<span style="font-size:11px;color:#666;display:inline-flex;'
                'align-items:center;gap:2px" title="Adjusted val. (handicap - weight + 55)">'
                'val.: <strong>' + adj_val_str + '</strong>' + val_circle + '</span>'
            )
        if draw != '—':
            meta_parts.append(
                '<span style="font-size:11px;color:#666;display:inline-flex;'
                'align-items:center;gap:4px">Draw: <strong>' + draw + '</strong>'
                + draw_pp_html + '</span>'
            )
        if daysince_html:
            meta_parts.append(daysince_html)

        # CHANGE 3: rtr (rating_after_race adjusted) shown in the name line with percentile circle
        if rtr_adj_str != '—':
            rtr_circle = _percentile_circle(rtr_adj_float, cat_rtr)
            meta_parts.append(
                '<span style="font-size:11px;color:#666;display:inline-flex;'
                'align-items:center;gap:2px" title="Adjusted rtr (rating_after_race - weight + 55)">'
                'rtr: <strong>' + rtr_adj_str + '</strong>' + rtr_circle + '</span>'
            )

        meta_html = ' '.join(meta_parts)
        form_context_table = _form_context_table_html(hid)

        card = (
            '<div style="margin-bottom:6px;">'
            '<div style="padding:10px 14px;border:1px solid ' + c_border + ';border-radius:5px;'
            'background:white;font-family:\'Helvetica Neue\',Arial,sans-serif;">'

            '<div style="margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid ' + c_border + '">'

            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px">'
            '<span style="font-weight:bold;color:' + c_navy + ';font-size:14px;'
            'margin-right:4px">' + str(horse) + '</span>'
            + meta_html
            + (' &nbsp;' + equip_html if equip_html else '')
            + (' &nbsp;' + headgear_change_html if headgear_change_html else '')
            + (' &nbsp;' + trainer_change_html if trainer_change_html else '')
            + (' &nbsp;' + owner_change_html if owner_change_html else '')
            + '</div>'

            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            + _odds_strip_html(horse)
            + '<span style="margin-left:auto">' + sp_html + '</span>'
            + '</div>'

            '</div>'

            '<div style="display:flex;gap:0;align-items:stretch;'
            'margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid ' + c_border + '">'
            '<div style="flex:0 0 auto;min-width:180px;padding-right:12px;'
            'border-right:1px solid ' + c_border + '">'
            + _badges_html(h_st, cat='horse')
            + horse_prefs_html
            + _horse_condition_panel_html(hid, horse_dist_grp, horse_going_grp)
            + '</div>'
            '<div style="flex:1 1 0;min-width:0;overflow-x:auto;padding-left:12px">'
            + (form_context_table if form_context_table else
               '<span style="font-size:10px;color:#bbb;font-style:italic">No form data</span>')
            + '</div>'
            '</div>'

            + '<div style="display:flex;gap:0;padding-top:2px">'
            + _entity_col('Trainer', trainer, t_st, cat='trainer',
                          prefs_html=trainer_prefs_html, border_right=True, hc_html=trainer_hc_html,
                          avg_pp=avg_pp_365_trainer.get(trainer if trainer != '—' else '', None),
                          all_pp=all_pp365_trainer)
            + _entity_col('Jockey',  jockey,  j_st, cat='jockey',
                          prefs_html=jockey_prefs_html,  border_right=True, hc_html=jockey_hc_html,
                          avg_pp=avg_pp_365_jockey.get(jockey if jockey != '—' else '', None),
                          all_pp=all_pp365_jockey)
            + _entity_col('Sire',    sire,    s_st, cat='sire',
                          prefs_html=sire_prefs_html,    border_right=False,
                          avg_pp=avg_pp_365_sire.get(sire if sire != '—' else '', None),
                          all_pp=all_pp365_sire)
            + '</div>'

            + '<!--VERDICT_START:' + _re_safe.sub(r'[^A-Za-z0-9]', '_', str(horse)) + '--><!--VERDICT_END:' + _re_safe.sub(r'[^A-Za-z0-9]', '_', str(horse)) + '-->'
            + '</div>'
            + '</div>'
        )

        # ── Collect JSON data (same computed vars, zero extra computation) ─────
        _ctx = horse_form_context.get(hid, [])
        _dp  = draw_pos_perc.get(str(draw_unique_val)) if draw_unique_val is not None else None
        def _st_dict(st):
            r2 = st.get('runs', 0)
            return {
                'runs':          r2,
                'wins':          st.get('wins', 0),
                'places':        st.get('places', 0),
                'win_pct':       round(100 * st['wins']   / r2, 1) if r2 else 0,
                'place_pct':     round(100 * st['places'] / r2, 1) if r2 else 0,
                'ae_place':      st.get('ae_place'),
                'prize_per_run': st.get('prizemoney'),
            }
        if 'saddle' in r.index and pd.notna(r.get('saddle')):
            _saddle_raw = r['saddle']
            try:
                _saddle = str(int(float(_saddle_raw)))
            except (ValueError, TypeError):
                _saddle = str(_saddle_raw)
        else:
            _saddle = None
        _race_json_horses.append({
            'name':                 horse,
            'saddle':               _saddle,
            'draw':                 draw,
            'draw_bias':            ({'bias': _dp[0], 'n': _dp[1]} if _dp else None),
            'age':                  (int(float(r['age'])) if 'age' in r.index and pd.notna(r.get('age')) else None),
            'sex':                  (sex if sex != '—' else None),
            'weight_kg':            (float(wt_raw) if wt_raw is not None and pd.notna(wt_raw) else None),
            'val':                  adj_val_float,
            'rtr':                  rtr_adj_float,
            'days_since_last_run':  horse_last_start.get(hid),
            'going_category_today': horse_going_grp,
            'distance_group_today': horse_dist_grp,
            'sp_tip':               (float(sp) if sp is not None and pd.notna(sp) else None),
            'blinkers':             (blink  if blink  not in ('', '—') else None),
            'hood':                 (hood_v if hood_v not in ('', '—') else None),
            'change_alerts':        _horse_change_alerts,
            'horse_stats':          _st_dict(h_st),
            'prefs':                _horse_prefs_data,
            'condition_panel':      horse_condition_stats.get(hid, {}),
            'trainer': {'name': trainer if trainer != '—' else '',
                        'pp365': avg_pp_365_trainer.get(trainer if trainer != '—' else '', None),
                        **_st_dict(t_st),
                        'prefs':    _trainer_prefs_data,
                        'hot_cold': hot_cold_trainer.get(trainer if trainer != '—' else '', None)},
            'jockey':  {'name': jockey  if jockey  != '—' else '',
                        'pp365': avg_pp_365_jockey.get(jockey if jockey != '—' else '', None),
                        **_st_dict(j_st),
                        'prefs':    _jockey_prefs_data,
                        'hot_cold': hot_cold_jockey.get(jockey if jockey != '—' else '', None)},
            'sire':    {'name': sire    if sire    != '—' else '',
                        'pp365': avg_pp_365_sire.get(sire if sire != '—' else '', None),
                        **_st_dict(s_st),
                        'prefs':    _sire_prefs_data},
            'recent_form': [
                {
                    'date':           e['date_str'],
                    'going_cat':      e.get('going_cat_raw', ''),
                    'dist_grp':       e.get('dist_grp_raw', ''),
                    'dist_m':         e.get('dist_m_raw'),
                    'pos':            e.get('ranking'),
                    'field':          e.get('runners'),
                    'lengths_beaten': e.get('lengths'),
                    'sp':             e.get('liveOdd'),
                    'arr':            e.get('arr_raw'),
                    'val':            e.get('val_raw'),
                    'jockey':         e.get('jockey_name', ''),
                    'notepad':        bool(notepad_flags and e.get('race_id') is not None
                                          and notepad_flags.get((str(e['race_id']), hid), False)),
                    'ff':             ff_stats_by_race.get(e.get('race_id')),
                    'meeting':        e.get('meeting'),
                    'class':          e.get('race_class'),
                    'type':           e.get('race_type'),
                    'opp':            [{'name': n, 'adj': a} for n, a in e.get('opp', [])],
                    'opp2':           [{'b_name': o2['b_name'], 'x_name': o2['x_name'],
                                        'score': o2['score']}
                                       for o2 in (e.get('opp2') or [])],
                }
                for e in _ctx
            ],
        })

        cards_html.append(card)

    _first_row = rows.iloc[0]
    _race_id = (_first_row['raceId'] if 'raceId' in rows.columns and pd.notna(_first_row.get('raceId')) else None)
    _race_json = {
        'raceId':         (str(int(_race_id)) if _race_id is not None else None),
        'meeting':        _s(_first_row, 'name_meeting'),
        'race':           _s(_first_row, 'name_race'),
        'going':          (_s(_first_row, 'going') if 'going' in rows.columns else ''),
        'going_category': today_going_grp or '',
        'distance_m':     (float(_first_row['distance']) if 'distance' in rows.columns and pd.notna(_first_row.get('distance')) else None),
        'distance_group': today_distance_grp or '',
        'race_type':      today_racetype or '',
        # total_prize_eur and race_class are added by export_all_races_html
        'field_size':     len(rows),
        'horses':         _race_json_horses,
    }

    _html_out = (
        '<div style="margin-bottom:14px;">'
        '<div style="font-size:11px;font-weight:bold;color:' + c_green + ';'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'
        'font-family:\'Helvetica Neue\',Arial,sans-serif;">'
        'Runners (' + str(len(rows)) + ')'
        '</div>'
        + ''.join(cards_html)
        + '</div>'
    )
    return _html_out, _race_json


def update_all_races_html_odds(output_dir, today_date, pmu_odds_history):
    """
    Fast odds-only update: reads existing HTML files produced by export_all_races_html,
    replaces only the <!--PMU_START:X-->...<!--PMU_END:X--> sections with fresh odds,
    and rewrites each file in-place.  Typically runs in seconds.
    """
    import os, re, glob

    date_str = today_date if isinstance(today_date, str) else today_date.strftime('%Y-%m-%d')

    # ── build horse_odds_strip (same logic as in _render_runners_html) ───────
    horse_odds_strip = {}
    if pmu_odds_history is not None and 'horseName' in pmu_odds_history.columns:
        _oh = pmu_odds_history.copy()
        _oh['_ts'] = pd.to_datetime(_oh['timestamp'], errors='coerce')
        _oh = _oh.dropna(subset=['_ts', 'odds', 'horseName'])
        _oh = _oh.sort_values('_ts', ascending=True)

        for _hname, _grp in _oh.groupby('horseName'):
            _odds_series = _grp['odds'].tolist()
            _ts_series   = _grp['_ts'].tolist()
            if not _odds_series:
                continue
            _deduped_idx = [0]
            for _i in range(1, len(_odds_series) - 1):
                if _odds_series[_i] != _odds_series[_i - 1]:
                    _deduped_idx.append(_i)
            if len(_odds_series) > 1:
                _deduped_idx.append(len(_odds_series) - 1)
            _deduped_idx  = sorted(set(_deduped_idx))
            _deduped_odds = [_odds_series[i] for i in _deduped_idx]
            _deduped_ts   = [_ts_series[i]   for i in _deduped_idx]
            if len(_deduped_odds) <= 5:
                _picked = list(range(len(_deduped_odds)))
            else:
                _interior_idx = list(range(1, len(_deduped_odds) - 1))
                _q = np.quantile(_interior_idx, [0.25, 0.5, 0.75])
                _q_positions  = sorted(set(int(round(q)) for q in _q))
                _picked = sorted(set([0] + _q_positions + [len(_deduped_odds) - 1]))
            horse_odds_strip[str(_hname)] = [
                (_deduped_ts[i], _deduped_odds[i]) for i in _picked
            ]

    # ── build the odds strip HTML for a single horse ─────────────────────────
    def _make_strip(horse_name):
        points = horse_odds_strip.get(str(horse_name), [])
        if not points:
            return ''
        chips = []
        prev_odds = None
        for ts, odds_val in points:
            try:
                odds_f = float(odds_val)
            except (TypeError, ValueError):
                continue
            if prev_odds is None:
                col, arrow = '#888', ''
            elif odds_f < prev_odds:
                col, arrow = '#1a7a3a', '▼'
            elif odds_f > prev_odds:
                col, arrow = '#c0392b', '▲'
            else:
                col, arrow = '#888', ''
            ts_str     = ts.strftime('%H:%M') if hasattr(ts, 'strftime') else str(ts)
            arrow_span = f'<span style="font-size:8px">{arrow}</span>' if arrow else ''
            chips.append(
                f'<span title="{ts_str}" style="white-space:nowrap;color:{col};font-size:10px">'
                f'{arrow_span}<strong style="color:{col}">{odds_f:.1f}</strong></span>'
            )
            prev_odds = odds_f
        if not chips:
            return ''
        joined = '<span style="color:#d0d4dc;font-size:9px;margin:0 2px">›</span>'.join(chips)
        return (
            f'<span style="display:inline-flex;align-items:center;gap:2px;'
            f'background:#f8f9fc;border:1px solid #e0e4ec;border-radius:8px;'
            f'padding:1px 7px;margin-left:4px">'
            f'<span style="font-size:9px;color:#aaa;margin-right:3px">PMU</span>'
            f'{joined}</span>'
        )

    # ── reverse-map safe_id → horse name ─────────────────────────────────────
    _safe_id_to_name = {re.sub(r'[^A-Za-z0-9]', '_', k): k for k in horse_odds_strip}

    def _replace_placeholder(m):
        safe_id    = m.group(1)
        horse_name = _safe_id_to_name.get(safe_id, safe_id)
        inner      = _make_strip(horse_name)
        return f'<!--PMU_START:{safe_id}-->{inner}<!--PMU_END:{safe_id}-->'

    _PMU_RE = re.compile(r'<!--PMU_START:([^-]+)-->.*?<!--PMU_END:\1-->', re.DOTALL)

    html_files = sorted(glob.glob(os.path.join(output_dir, f'{date_str}__*.html')))
    if not html_files:
        print(f'⚠  No HTML files found for {date_str} in {output_dir}')
        return []

    updated = []
    for fpath in html_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = _PMU_RE.sub(_replace_placeholder, content)
        if new_content != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        updated.append(os.path.basename(fpath))

    print(f'✅ PMU odds updated in {len(updated)} files')
    return updated


def update_verdicts_in_html(output_dir, today_date, horse_verdicts):
    """
    Inject AI verdicts into <!--VERDICT_START:X-->...<!--VERDICT_END:X--> placeholders
    in existing HTML files.  Analogous to update_all_races_html_odds for PMU odds.
    Rewrites each file in-place; only touches the verdict sections.
    """
    import os, re, glob

    date_str = today_date if isinstance(today_date, str) else today_date.strftime('%Y-%m-%d')

    def _verdict_html(verdict_text):
        return (
            '<div style="margin-top:8px;padding:7px 12px;'
            'background:#f5f7ff;border-left:3px solid #5b8dd9;'
            'border-radius:0 4px 4px 0;font-size:12px;color:#333;'
            'line-height:1.55;font-family:\'Helvetica Neue\',Arial,sans-serif">'
            '<span style="font-size:9px;font-weight:bold;color:#5b8dd9;'
            'text-transform:uppercase;letter-spacing:.07em;display:block;margin-bottom:3px">'
            'AI Verdict</span>'
            + str(verdict_text)
            + '</div>'
        )

    # Build safe_id → verdict HTML mapping
    _safe_id_to_verdict = {}
    for horse_name, verdict_text in (horse_verdicts or {}).items():
        sid = re.sub(r'[^A-Za-z0-9]', '_', str(horse_name))
        _safe_id_to_verdict[sid] = verdict_text

    _VERDICT_RE = re.compile(
        r'<!--VERDICT_START:([^-]+)-->.*?<!--VERDICT_END:\1-->', re.DOTALL
    )

    def _replace_verdict(m):
        sid   = m.group(1)
        vtext = _safe_id_to_verdict.get(sid)
        inner = _verdict_html(vtext) if vtext else ''
        return f'<!--VERDICT_START:{sid}-->{inner}<!--VERDICT_END:{sid}-->'

    updated = []
    for fpath in sorted(glob.glob(os.path.join(output_dir, f'{date_str}__*.html'))):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = _VERDICT_RE.sub(_replace_verdict, content)
        if new_content != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            updated.append(os.path.basename(fpath))

    print(f'✅ AI verdicts injected into {len(updated)} files')
    return updated


def update_race_verdicts_in_html(output_dir, today_date, race_verdicts):
    """
    Inject race-level NAP/EW verdicts into <!--RACE_VERDICT:key-->...<!--RACE_VERDICT_END:key-->
    placeholders in existing HTML files.  race_verdicts = {race_key: verdict_dict}.
    """
    import os, re, glob

    date_str = today_date if isinstance(today_date, str) else today_date.strftime('%Y-%m-%d')

    _key_to_html = {}
    for race_key, verdict in (race_verdicts or {}).items():
        sid = re.sub(r'[^A-Za-z0-9]', '_', str(race_key))
        _key_to_html[sid] = _render_race_verdict_html(verdict)

    _RV_RE = re.compile(
        r'<!--RACE_VERDICT:([^-]+)-->.*?<!--RACE_VERDICT_END:\1-->', re.DOTALL
    )

    def _replace_rv(m):
        sid  = m.group(1)
        html = _key_to_html.get(sid, '')
        return f'<!--RACE_VERDICT:{sid}-->{html}<!--RACE_VERDICT_END:{sid}-->'

    updated = []
    for fpath in sorted(glob.glob(os.path.join(output_dir, f'{date_str}__*.html'))):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = _RV_RE.sub(_replace_rv, content)
        if new_content != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            updated.append(os.path.basename(fpath))

    print(f'✅ Race verdicts injected into {len(updated)} files')
    return updated
