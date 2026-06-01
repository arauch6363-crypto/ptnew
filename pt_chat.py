#!/usr/bin/env python3
"""
PT Chat — interactive horse racing analyst chatbot.

Queries your parquet history to answer questions about horses, trainers,
jockeys, and upcoming races with real, up-to-date data.

Usage (terminal):
    python pt_chat.py
    python pt_chat.py --base /content/drive/MyDrive/PT
    python pt_chat.py --model claude-haiku-4-5-20251001   # cheaper

Usage (Colab cell):
    from pt_chat import PTChat, load_data
    data = load_data('/content/drive/MyDrive/PT')
    PTChat(data).run()
"""

import argparse
import difflib
import json
import os
import sys
from datetime import date, timedelta

import anthropic
import numpy as np
import pandas as pd

TODAY = date.today().isoformat()
_365_AGO = (date.today() - timedelta(days=365)).isoformat()
_21_AGO  = (date.today() - timedelta(days=21)).isoformat()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an elite horse racing analyst specialising in French flat racing,
writing in the authoritative, data-driven style of the Racing Post.

You have access to tools that fetch REAL data from a historical database of
30 000+ races. Always call the appropriate tool before making claims about a
specific horse, trainer, jockey, or race. Never guess statistics.

SIGNAL VOCABULARY
─────────────────
ARR          Adjusted Racing Rating — performance vs the field on the day;
             compare WITHIN a race, not in absolute terms.
             A horse with arr_max 3+ points above field median has a real edge.
pos_perc     Normalised finishing position (1 = won, 0 = last).
pp365        Mean pos_perc over the last 365 days — overall quality of a trainer
             or jockey. > 0.5 is strong; 0.55+ is elite.
hot/cold     pp365 of last 21 days vs pp365. Hot = stable in form; Cold = below form.
ae_place     Actual places / expected places from market odds. > 1.0 outperforms.
             Unreliable below 20 runs.
val          Handicap rating edge within the field.
sp / liveOdd Starting price (lower = market favourite).
going_cat    VERY SLOW | SLOW | FAST | VERY FAST | PSF
dist_group   0-1200 | 1201-1600 | 1601-2200 | 2201-2600 | >2600  (metres)
race_type    H = handicap | R = claimer | M = maiden | None = conditions

FRENCH RACING CONTEXT
─────────────────────
Major flat venues: Longchamp, Chantilly, Deauville, Saint-Cloud, Maisons-Laffitte,
Vincennes (trot only — ignore), Cagnes-sur-Mer (winter), Lyon-Parilly, Bordeaux.
Season peaks: spring (April–June Longchamp/Chantilly), summer (Deauville August),
autumn Classics (Arc week, October, Longchamp).
Weights in kg; French handicap system rates horses 0–115.

HOW TO ANALYSE A RACE
──────────────────────
1. Fetch the field with get_today_race or get_horse_profile for each contender.
2. Compare ARR stats within the field — who has the highest arr_max / arr_median?
3. Check going and distance suitability using each horse's record.
4. Check trainer and jockey form (pp365, hot/cold).
5. Identify market edge: which strong signals are on a horse with a generous SP?
6. Give a clear NAP (best win) and each-way recommendation with concise reasoning.

Be direct. Back every claim with a number from the tools. Acknowledge uncertainty.
"""

# ── Data loading ───────────────────────────────────────────────────────────────

class PTData:
    def __init__(self):
        self.runners: pd.DataFrame = pd.DataFrame()
        self.races:   pd.DataFrame = pd.DataFrame()
        self.runners_tdy: pd.DataFrame | None = None
        self.races_tdy:   pd.DataFrame | None = None


def load_data(base: str) -> PTData:
    d = PTData()

    def _load(name):
        path = os.path.join(base, name)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if 'date' in df.columns:
                df['_dt'] = pd.to_datetime(df['date'], errors='coerce')
            print(f"  ✓ {name}: {len(df):,} rows")
            return df
        print(f"  – {name}: not found")
        return None

    print(f"Loading data from {base} ...")
    r = _load('runners.parquet')
    d.runners = r if r is not None else pd.DataFrame()

    rc = _load('races.parquet')
    d.races = rc if rc is not None else pd.DataFrame()

    d.runners_tdy = _load('runners_tdy.parquet')
    d.races_tdy   = _load('races_tdy.parquet')

    print(f"Ready. {len(d.runners):,} historical runner records loaded.\n")
    return d


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _fuzzy(candidates: list[str], query: str, n: int = 5, cutoff: float = 0.4) -> list[str]:
    return difflib.get_close_matches(query, candidates, n=n, cutoff=cutoff)


def _career_stats(grp: pd.DataFrame) -> dict:
    runs   = len(grp)
    wins   = int(grp['win'].sum()) if 'win' in grp.columns else 0
    places = int(grp['place'].sum()) if 'place' in grp.columns else 0
    prize  = int(grp['prizemoney'].sum()) if 'prizemoney' in grp.columns else 0
    return {
        'runs': runs,
        'wins': wins,
        'places': places,
        'win_pct':   round(100 * wins / runs, 1) if runs else 0,
        'place_pct': round(100 * places / runs, 1) if runs else 0,
        'prize_per_run': round(prize / runs) if runs else 0,
    }


def _arr_stats(grp: pd.DataFrame) -> dict:
    if 'ARR' not in grp.columns:
        return {}
    arr = grp['ARR'].dropna()
    if arr.empty:
        return {}
    recent = arr.head(5).values
    trend  = None
    if len(recent) >= 3:
        slope = float(np.polyfit(range(len(recent)), recent[::-1], 1)[0])
        trend = f"{slope:+.1f}/run"
    return {
        'avg':    round(float(arr.mean()), 1),
        'median': round(float(arr.median()), 1),
        'max':    round(float(arr.max()), 1),
        'trend':  trend,
    }


def _record_by(grp: pd.DataFrame, col: str) -> dict:
    if col not in grp.columns:
        return {}
    out = {}
    for val, sub in grp.groupby(col):
        if pd.isna(val):
            continue
        runs   = len(sub)
        wins   = int(sub['win'].sum()) if 'win' in sub.columns else 0
        places = int(sub['place'].sum()) if 'place' in sub.columns else 0
        out[str(val)] = {
            'runs':  runs,
            'wins':  wins,
            'places': places,
            'win_pct':   round(100 * wins / runs, 1) if runs else 0,
            'place_pct': round(100 * places / runs, 1) if runs else 0,
        }
    return out


def _pp365(grp: pd.DataFrame, days: int = 365) -> float | None:
    if 'pos_perc' not in grp.columns or '_dt' not in grp.columns:
        return None
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    sub = grp[grp['_dt'] >= cutoff]['pos_perc'].dropna()
    return round(float(sub.mean()), 3) if not sub.empty else None


def _recent_form(grp: pd.DataFrame, races_df: pd.DataFrame, n: int = 12) -> list[dict]:
    if '_dt' not in grp.columns:
        return []
    sub = grp.sort_values('_dt', ascending=False).head(n)
    rows = []
    for _, r in sub.iterrows():
        entry: dict = {}
        for col in ('date', 'going_category', 'distance_group', 'position', 'ARR',
                    'liveOdd', 'jockeyName', 'comment', 'win', 'place', 'prizemoney'):
            if col in r.index and pd.notna(r[col]):
                entry[col] = r[col]

        # Enrich from races table
        rid = r.get('raceId')
        if rid is not None and not races_df.empty and 'raceId' in races_df.columns:
            race_row = races_df[races_df['raceId'] == rid]
            if not race_row.empty:
                rr = race_row.iloc[0]
                for col in ('meeting', 'race', 'race_type', 'race_class',
                            'total_prize_eur', 'distance_m'):
                    if col in rr.index and pd.notna(rr[col]):
                        entry[col] = rr[col]
                # field size
                if 'field_size' not in entry:
                    fs = races_df[races_df['raceId'] == rid]
                    if not fs.empty and 'field_size' in fs.columns:
                        entry['field_size'] = int(fs['field_size'].iloc[0])

        rows.append(entry)
    return rows


# ── Tool implementations ───────────────────────────────────────────────────────

def tool_search_horses(data: PTData, query: str) -> dict:
    if data.runners.empty or 'horseName' not in data.runners.columns:
        return {'error': 'No runner data loaded'}
    names = data.runners['horseName'].dropna().unique().tolist()
    matches = _fuzzy(names, query, n=8, cutoff=0.35)
    return {'query': query, 'matches': matches}


def tool_get_horse_profile(data: PTData, horse_name: str) -> dict:
    if data.runners.empty or 'horseName' not in data.runners.columns:
        return {'error': 'No runner data loaded'}

    grp = data.runners[data.runners['horseName'] == horse_name]
    if grp.empty:
        names = data.runners['horseName'].dropna().unique().tolist()
        suggestions = _fuzzy(names, horse_name)
        return {'error': f'Horse "{horse_name}" not found', 'suggestions': suggestions}

    grp = grp.sort_values('_dt', ascending=False) if '_dt' in grp.columns else grp

    last = grp.iloc[0]
    days_since = None
    if '_dt' in grp.columns and not grp['_dt'].isna().all():
        last_dt = grp['_dt'].dropna().max()
        days_since = (date.today() - last_dt.date()).days

    trainer = str(last.get('trainerName', '')) if pd.notna(last.get('trainerName')) else None
    jockey  = str(last.get('jockeyName', ''))  if pd.notna(last.get('jockeyName'))  else None
    sire    = str(last.get('horseSir', ''))     if pd.notna(last.get('horseSir'))    else None

    # Trainer pp365
    trainer_pp = None
    trainer_pp_21 = None
    if trainer and 'trainerName' in data.runners.columns:
        t_grp = data.runners[data.runners['trainerName'] == trainer]
        trainer_pp    = _pp365(t_grp, 365)
        trainer_pp_21 = _pp365(t_grp, 21)

    # Jockey pp365
    jockey_pp = None
    jockey_pp_21 = None
    if jockey and 'jockeyName' in data.runners.columns:
        j_grp = data.runners[data.runners['jockeyName'] == jockey]
        jockey_pp    = _pp365(j_grp, 365)
        jockey_pp_21 = _pp365(j_grp, 21)

    profile = {
        'name':    horse_name,
        'age':     int(last['age']) if 'age' in last.index and pd.notna(last['age']) else None,
        'sex':     str(last['sex']) if 'sex' in last.index and pd.notna(last['sex']) else None,
        'sire':    sire,
        'trainer': trainer,
        'trainer_pp365': trainer_pp,
        'trainer_pp21':  trainer_pp_21,
        'trainer_hot':   (trainer_pp_21 > trainer_pp) if (trainer_pp and trainer_pp_21) else None,
        'jockey_recent': jockey,
        'jockey_pp365':  jockey_pp,
        'jockey_pp21':   jockey_pp_21,
        'days_since_last_run': days_since,
        'career':           _career_stats(grp),
        'arr_stats':        _arr_stats(grp),
        'going_record':     _record_by(grp, 'going_category'),
        'distance_record':  _record_by(grp, 'distance_group'),
        'recent_form':      _recent_form(grp, data.races, n=12),
    }
    return profile


def tool_get_trainer_profile(data: PTData, trainer_name: str) -> dict:
    if data.runners.empty or 'trainerName' not in data.runners.columns:
        return {'error': 'No runner data loaded'}

    grp = data.runners[data.runners['trainerName'] == trainer_name]
    if grp.empty:
        names = data.runners['trainerName'].dropna().unique().tolist()
        suggestions = _fuzzy(names, trainer_name)
        return {'error': f'Trainer "{trainer_name}" not found', 'suggestions': suggestions}

    pp365 = _pp365(grp, 365)
    pp21  = _pp365(grp, 21)

    # Top jockeys (by runs)
    top_jockeys = []
    if 'jockeyName' in grp.columns:
        jk = (grp.groupby('jockeyName')
                  .agg(runs=('win', 'count'), wins=('win', 'sum'))
                  .sort_values('runs', ascending=False)
                  .head(5))
        for name, row in jk.iterrows():
            top_jockeys.append({'name': str(name), 'runs': int(row['runs']),
                                 'wins': int(row['wins'])})

    # Top meetings
    top_meetings = []
    if 'raceId' in grp.columns and not data.races.empty and 'raceId' in data.races.columns:
        merged = grp.merge(data.races[['raceId', 'meeting']], on='raceId', how='left')
        if 'meeting' in merged.columns:
            mt = (merged.groupby('meeting')
                        .agg(runs=('win', 'count'), wins=('win', 'sum'))
                        .sort_values('wins', ascending=False)
                        .head(5))
            for name, row in mt.iterrows():
                top_meetings.append({'meeting': str(name), 'runs': int(row['runs']),
                                      'wins': int(row['wins'])})

    # Recent results (last 10 runs)
    recent = []
    if '_dt' in grp.columns:
        r10 = grp.sort_values('_dt', ascending=False).head(10)
        for _, row in r10.iterrows():
            entry = {'horse': row.get('horseName'), 'date': str(row.get('date', '')),
                     'win': bool(row.get('win', False)), 'place': bool(row.get('place', False))}
            recent.append(entry)

    return {
        'name':     trainer_name,
        'pp365':    pp365,
        'pp21':     pp21,
        'hot':      (pp21 > pp365) if (pp365 and pp21) else None,
        'career':   _career_stats(grp),
        'top_jockeys':  top_jockeys,
        'top_meetings': top_meetings,
        'recent_10':    recent,
    }


def tool_get_jockey_profile(data: PTData, jockey_name: str) -> dict:
    if data.runners.empty or 'jockeyName' not in data.runners.columns:
        return {'error': 'No runner data loaded'}

    grp = data.runners[data.runners['jockeyName'] == jockey_name]
    if grp.empty:
        names = data.runners['jockeyName'].dropna().unique().tolist()
        suggestions = _fuzzy(names, jockey_name)
        return {'error': f'Jockey "{jockey_name}" not found', 'suggestions': suggestions}

    pp365 = _pp365(grp, 365)
    pp21  = _pp365(grp, 21)

    top_trainers = []
    if 'trainerName' in grp.columns:
        tk = (grp.groupby('trainerName')
                  .agg(runs=('win', 'count'), wins=('win', 'sum'))
                  .sort_values('wins', ascending=False)
                  .head(5))
        for name, row in tk.iterrows():
            top_trainers.append({'name': str(name), 'runs': int(row['runs']),
                                  'wins': int(row['wins'])})

    recent = []
    if '_dt' in grp.columns:
        r10 = grp.sort_values('_dt', ascending=False).head(10)
        for _, row in r10.iterrows():
            recent.append({'horse': row.get('horseName'), 'date': str(row.get('date', '')),
                           'win': bool(row.get('win', False))})

    return {
        'name':   jockey_name,
        'pp365':  pp365,
        'pp21':   pp21,
        'hot':    (pp21 > pp365) if (pp365 and pp21) else None,
        'career': _career_stats(grp),
        'top_trainers': top_trainers,
        'recent_10':    recent,
    }


def tool_get_today_race(data: PTData, query: str) -> dict:
    """Find today's race(s) matching a meeting name or race name fragment."""
    if data.runners_tdy is None:
        return {'error': "No today's data loaded — run PT_getData first"}

    df = data.runners_tdy.copy()
    query_lower = query.lower()

    # Try matching name_meeting or name_race
    mask = pd.Series(False, index=df.index)
    for col in ('name_meeting', 'name_race', 'meeting', 'race'):
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False)

    matched = df[mask]
    if matched.empty:
        available = []
        for col in ('name_meeting', 'meeting'):
            if col in df.columns:
                available = df[col].dropna().unique().tolist()
                break
        return {'error': f'No race matching "{query}" today', 'available_meetings': available}

    races_out = []
    race_col = 'name_race' if 'name_race' in matched.columns else 'race'
    meeting_col = 'name_meeting' if 'name_meeting' in matched.columns else 'meeting'

    for race_name, race_grp in matched.groupby(race_col):
        horses = []
        for _, row in race_grp.iterrows():
            h = {'name': row.get('horseName'), 'saddle': row.get('saddle')}
            if 'SP' in row.index and pd.notna(row['SP']):
                h['sp_tip'] = round(float(row['SP']), 1)
            if 'liveOdd' in row.index and pd.notna(row['liveOdd']):
                h['live_odd'] = round(float(row['liveOdd']), 1)
            if 'age' in row.index and pd.notna(row['age']):
                h['age'] = int(row['age'])
            if 'trainerName' in row.index:
                h['trainer'] = row['trainerName']
            if 'jockeyName' in row.index:
                h['jockey'] = row['jockeyName']
            horses.append(h)

        race_info: dict = {
            'meeting': str(race_grp[meeting_col].iloc[0]) if meeting_col in race_grp.columns else query,
            'race':    str(race_name),
            'field':   horses,
        }
        # Race metadata from races_tdy
        if data.races_tdy is not None and 'raceId' in race_grp.columns:
            rids = race_grp['raceId'].dropna().unique()
            if len(rids):
                rm = data.races_tdy[data.races_tdy['raceId'].isin(rids)]
                if not rm.empty:
                    rr = rm.iloc[0]
                    for col in ('going', 'going_category', 'distance_m', 'race_type',
                                'race_class', 'total_prize_eur'):
                        if col in rr.index and pd.notna(rr[col]):
                            race_info[col] = rr[col]
        races_out.append(race_info)

    return {'query': query, 'races': races_out}


def tool_get_head_to_head(data: PTData, horse1: str, horse2: str) -> dict:
    if data.runners.empty or 'horseName' not in data.runners.columns:
        return {'error': 'No runner data loaded'}

    grp1 = data.runners[data.runners['horseName'] == horse1]
    grp2 = data.runners[data.runners['horseName'] == horse2]

    if grp1.empty:
        return {'error': f'Horse "{horse1}" not found'}
    if grp2.empty:
        return {'error': f'Horse "{horse2}" not found'}

    if 'raceId' not in data.runners.columns:
        return {'error': 'raceId column missing — cannot find shared races'}

    shared_ids = set(grp1['raceId'].dropna()) & set(grp2['raceId'].dropna())
    if not shared_ids:
        return {'horse1': horse1, 'horse2': horse2, 'shared_races': 0,
                'note': 'These horses have never met in the same race'}

    meetings = []
    for rid in sorted(shared_ids):
        r1 = grp1[grp1['raceId'] == rid]
        r2 = grp2[grp2['raceId'] == rid]
        if r1.empty or r2.empty:
            continue
        r1r = r1.iloc[0]
        r2r = r2.iloc[0]
        entry: dict = {'raceId': rid,
                       'date': str(r1r.get('date', '')),
                       horse1: {'pos': r1r.get('position'), 'arr': r1r.get('ARR'),
                                'sp': r1r.get('liveOdd')},
                       horse2: {'pos': r2r.get('position'), 'arr': r2r.get('ARR'),
                                'sp': r2r.get('liveOdd')}}
        if not data.races.empty and 'raceId' in data.races.columns:
            rm = data.races[data.races['raceId'] == rid]
            if not rm.empty:
                rr = rm.iloc[0]
                entry['meeting'] = rr.get('meeting')
                entry['going_category'] = rr.get('going_category')
        meetings.append(entry)

    meetings.sort(key=lambda x: x.get('date', ''), reverse=True)
    h1_wins = sum(1 for m in meetings
                  if m.get(horse1, {}).get('pos') == 1 or
                     (m.get(horse1, {}).get('pos') or 99) < (m.get(horse2, {}).get('pos') or 99))
    return {
        'horse1': horse1, 'horse2': horse2,
        'shared_races': len(meetings),
        f'{horse1}_better': h1_wins,
        f'{horse2}_better': len(meetings) - h1_wins,
        'meetings': meetings[:10],
    }


# ── Anthropic tool definitions ─────────────────────────────────────────────────

TOOL_DEFS = [
    {
        'name': 'search_horses',
        'description': 'Fuzzy-search horse names in the database. Use this first if unsure of exact spelling.',
        'input_schema': {
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': 'Horse name or partial name'}},
            'required': ['query'],
        },
    },
    {
        'name': 'get_horse_profile',
        'description': (
            'Full profile for a horse: career stats, recent form (last 12 runs), '
            'ARR stats, going/distance record, trainer and jockey form.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {'horse_name': {'type': 'string', 'description': 'Exact horse name'}},
            'required': ['horse_name'],
        },
    },
    {
        'name': 'get_trainer_profile',
        'description': 'Trainer stats: pp365, hot/cold form, top jockeys, top meetings, recent results.',
        'input_schema': {
            'type': 'object',
            'properties': {'trainer_name': {'type': 'string'}},
            'required': ['trainer_name'],
        },
    },
    {
        'name': 'get_jockey_profile',
        'description': 'Jockey stats: pp365, hot/cold form, top trainers, recent results.',
        'input_schema': {
            'type': 'object',
            'properties': {'jockey_name': {'type': 'string'}},
            'required': ['jockey_name'],
        },
    },
    {
        'name': 'get_today_race',
        'description': (
            "Find today's race(s) by meeting name or race name. "
            "Returns the full field with SP tips, live odds, trainer/jockey."
        ),
        'input_schema': {
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': 'Meeting or race name fragment'}},
            'required': ['query'],
        },
    },
    {
        'name': 'get_head_to_head',
        'description': 'Find all races where two horses competed in the same field and compare their results.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'horse1': {'type': 'string'},
                'horse2': {'type': 'string'},
            },
            'required': ['horse1', 'horse2'],
        },
    },
]

_TOOL_FNS = {
    'search_horses':      lambda d, inp: tool_search_horses(d, **inp),
    'get_horse_profile':  lambda d, inp: tool_get_horse_profile(d, **inp),
    'get_trainer_profile':lambda d, inp: tool_get_trainer_profile(d, **inp),
    'get_jockey_profile': lambda d, inp: tool_get_jockey_profile(d, **inp),
    'get_today_race':     lambda d, inp: tool_get_today_race(d, **inp),
    'get_head_to_head':   lambda d, inp: tool_get_head_to_head(d, **inp),
}


# ── Chat engine ────────────────────────────────────────────────────────────────

class PTChat:
    def __init__(self, data: PTData, api_key: str | None = None,
                 model: str = 'claude-sonnet-4-6'):
        self.data     = data
        self.model    = model
        self.client   = anthropic.Anthropic(
            api_key=api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        )
        self.messages: list[dict] = []
        self._system_block = [
            {'type': 'text', 'text': SYSTEM_PROMPT, 'cache_control': {'type': 'ephemeral'}}
        ]

    def _dispatch(self, name: str, inp: dict) -> str:
        fn = _TOOL_FNS.get(name)
        if fn is None:
            return json.dumps({'error': f'Unknown tool: {name}'})
        try:
            result = fn(self.data, inp)
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'error': str(exc)})

    def turn(self, user_text: str) -> str:
        """Process one user turn; return assistant reply text."""
        self.messages.append({'role': 'user', 'content': user_text})

        while True:
            resp = self.client.messages.create(
                model=self.model,
                system=self._system_block,
                tools=TOOL_DEFS,
                messages=self.messages,
                max_tokens=4096,
            )

            # Append assistant message
            self.messages.append({'role': 'assistant', 'content': resp.content})

            if resp.stop_reason == 'end_turn':
                return next(
                    (b.text for b in resp.content if hasattr(b, 'text')), ''
                )

            if resp.stop_reason == 'tool_use':
                tool_results = []
                for block in resp.content:
                    if block.type != 'tool_use':
                        continue
                    result_str = self._dispatch(block.name, block.input)
                    tool_results.append({
                        'type':        'tool_result',
                        'tool_use_id': block.id,
                        'content':     result_str,
                    })
                self.messages.append({'role': 'user', 'content': tool_results})
            else:
                # Unexpected stop reason
                return next(
                    (b.text for b in resp.content if hasattr(b, 'text')), ''
                )

    def run(self):
        """Interactive REPL loop."""
        print("PT Racing Analyst — type your question, 'quit' to exit, 'clear' to reset.\n")
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not user:
                continue
            if user.lower() in ('quit', 'exit', 'q'):
                print("Goodbye.")
                break
            if user.lower() == 'clear':
                self.messages = []
                print("Conversation cleared.\n")
                continue

            try:
                reply = self.turn(user)
                print(f"\nAnalyst: {reply}\n")
            except anthropic.APIError as exc:
                print(f"API error: {exc}\n")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PT Racing Analyst chatbot')
    parser.add_argument('--base',  default=os.environ.get('PT_BASE', '/content/drive/MyDrive/PT'),
                        help='Path to PT data directory')
    parser.add_argument('--model', default='claude-sonnet-4-6',
                        help='Claude model to use')
    parser.add_argument('--api-key', default=None, help='Anthropic API key (or set env var)')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY or pass --api-key")
        sys.exit(1)

    data = load_data(args.base)
    PTChat(data, api_key=api_key, model=args.model).run()


if __name__ == '__main__':
    main()
