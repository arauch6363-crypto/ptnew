#!/usr/bin/env python3
"""
PT HTML Fast — Railway runner

Fetches live PMU odds and exports HTML race cards to Google Drive.
Replaces PT_Create_HTMLs_fast.ipynb for automated / non-Colab execution.

Required environment variables:
    GITHUB_TOKEN         Read odds_{TODAY}.json from pmu-tracker repo
    GOOGLE_CREDENTIALS   Service account JSON (as a string) with Drive access

Optional:
    ANTHROPIC_API_KEY    Needed only if compute_notepad_flags is called here
                         (it runs in PT_Vorarbeiten, not here)
"""

import io
import json
import os
import pickle
import re
import sys
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ── Constants ─────────────────────────────────────────────────────────────────
BERLIN       = pytz.timezone('Europe/Berlin')
TODAY        = datetime.now(BERLIN).strftime('%Y-%m-%d')
REPO_ROOT    = Path(__file__).parent.parent
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
PMU_REPO     = 'arauch6363-crypto/pmu-tracker'
DRIVE_FOLDER = 'PT'


# ── Google Drive helpers ──────────────────────────────────────────────────────
def _drive_service():
    creds_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=['https://www.googleapis.com/auth/drive'],
    )
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def _find(svc, name: str, parent_id: str) -> str | None:
    q = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    res = svc.files().list(q=q, fields='files(id)').execute()
    files = res.get('files', [])
    return files[0]['id'] if files else None


def _download(svc, file_id: str, dest: Path) -> None:
    req = svc.files().get_media(fileId=file_id)
    with dest.open('wb') as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()


def _upload(svc, local: Path, name: str, parent_id: str) -> None:
    media = MediaFileUpload(str(local), resumable=False)
    existing = _find(svc, name, parent_id)
    if existing:
        svc.files().update(fileId=existing, media_body=media).execute()
    else:
        svc.files().create(
            body={'name': name, 'parents': [parent_id]},
            media_body=media,
        ).execute()


def _find_folder(svc, name: str, parent_id: str | None = None) -> str:
    q = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = svc.files().list(q=q, fields='files(id)').execute()
    files = res.get('files', [])
    if not files:
        raise FileNotFoundError(
            f"Drive folder '{name}' not found."
            + (" Share it with the service account email." if not parent_id else "")
        )
    return files[0]['id']


def _get_or_create_folder(svc, name: str, parent_id: str) -> str:
    existing = _find(svc, name, parent_id)
    if existing:
        return existing
    f = svc.files().create(
        body={
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id],
        },
        fields='id',
    ).execute()
    return f['id']


# ── PMU odds ──────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    name = unicodedata.normalize('NFD', str(name).strip())
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return re.sub(r'^#\d+\s+', '', name).upper().strip()


def _filter_before_race(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False)
    race_start = pd.to_datetime(
        df['timestamp'].dt.date.astype(str) + ' ' + df['heure']
    )
    return df[df['timestamp'] < race_start]


def fetch_pmu_odds(runners_tdy: pd.DataFrame) -> pd.DataFrame:
    url = (
        f'https://raw.githubusercontent.com/{PMU_REPO}'
        f'/main/history/odds_{TODAY}.json'
    )
    resp = requests.get(url, headers={'Authorization': f'token {GITHUB_TOKEN}'}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = [
        {
            'timestamp': ts,
            'race': rk,
            'hippodrome': race['hippodrome'],
            'heure': race['heure'],
            'horse': horse,
            'odds': d['odds'],
            'tendance': d['tendance'],
            'magnitude': d['magnitude'],
            'favoris': d['favoris'],
        }
        for ts, snapshot in data.items()
        for rk, race in snapshot.items()
        for horse, d in race['horses'].items()
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values(['race', 'heure', 'horse', 'timestamp'])
        .reset_index(drop=True)
    )
    df = _filter_before_race(df)
    df['horse'] = df['horse'].apply(normalize_name)

    runners_tdy = runners_tdy.copy()
    runners_tdy['_key'] = runners_tdy['horseName'].apply(normalize_name)
    df = (
        df.merge(
            runners_tdy[['_key', 'horseName']],
            left_on='horse',
            right_on='_key',
            how='left',
        )
        .drop(columns=['_key', 'horse'])
        .dropna(subset=['horseName'])
        .reset_index(drop=True)
    )
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f'PT HTML Fast  |  {TODAY}  |  Berlin {datetime.now(BERLIN).strftime("%H:%M")}')

    svc    = _drive_service()
    pt_id  = _find_folder(svc, DRIVE_FOLDER)
    races_id = _get_or_create_folder(svc, 'races', pt_id)
    print(f'✓ Drive: {DRIVE_FOLDER}/ found')

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        # ── Download pre-computed files from Drive ──────────────────────────
        needed = [
            'runners_processed.parquet',
            'df_with_ratings.parquet',
            f'notepad_flags_{TODAY}.pkl',
            f'precomputed_tdy_{TODAY}.pkl',
        ]
        print('Downloading from Drive...')
        for fname in needed:
            fid = _find(svc, fname, pt_id)
            if not fid:
                raise FileNotFoundError(
                    f'{fname} not found in Drive. '
                    'Run PT_Vorarbeiten first.'
                )
            _download(svc, fid, tmp / fname)
            print(f'  ✓ {fname}')

        # ── Load DataFrames ─────────────────────────────────────────────────
        runners         = pd.read_parquet(tmp / 'runners_processed.parquet')
        df_with_ratings = pd.read_parquet(tmp / 'df_with_ratings.parquet')

        with open(tmp / f'notepad_flags_{TODAY}.pkl', 'rb') as f:
            notepad_flags = pickle.load(f)

        with open(tmp / f'precomputed_tdy_{TODAY}.pkl', 'rb') as f:
            tdy = pickle.load(f)

        runners_tdy = tdy['runners_tdy']
        webTips_tdy = tdy['webTips_tdy']
        races_tdy   = tdy['races_tdy']
        today_tips  = tdy['today_tips']

        print(
            f'✓ {len(runners):,} hist runners | '
            f'{len(runners_tdy)} today runners across '
            f'{runners_tdy["raceId"].nunique()} races'
        )

        # ── Fetch live PMU odds ─────────────────────────────────────────────
        print('Fetching PMU odds from GitHub...')
        pmu_odds_history = fetch_pmu_odds(runners_tdy)
        print(f'✓ {len(pmu_odds_history):,} odds rows')

        # ── Filter to upcoming races (Berlin time) ──────────────────────────
        now_str = datetime.now(BERLIN).strftime('%H:%M:%S')
        races_tdy['time'] = races_tdy['time'].astype(str)
        races_filt   = races_tdy[races_tdy['time'] > now_str].copy().reset_index(drop=True)
        runners_filt = (
            runners_tdy[runners_tdy['id_race'].isin(races_filt['id_race'])]
            .copy()
            .reset_index(drop=True)
        )
        print(f'✓ {len(races_filt)} upcoming races (after {now_str} Berlin)')

        if races_filt.empty:
            print('No upcoming races — nothing to export.')
            return

        # ── Load shared HTML module ─────────────────────────────────────────
        sys.path.insert(0, str(REPO_ROOT))
        from pt_html_functions import export_all_races_html  # noqa: E402

        # ── Export HTMLs ────────────────────────────────────────────────────
        out_dir = tmp / 'races'
        out_dir.mkdir()

        exported = export_all_races_html(
            df_hist          = runners,
            df_today         = runners_filt,
            webTips_tdy      = webTips_tdy,
            today_tips       = today_tips,
            races_tdy        = races_filt,
            df_with_ratings  = df_with_ratings,
            notepad_flags    = notepad_flags,
            pmu_odds_history = pmu_odds_history,
            output_dir       = str(out_dir),
        )

        # ── Upload HTMLs to Drive ───────────────────────────────────────────
        print(f'Uploading {len(exported)} HTML files to Drive...')
        for fname in exported:
            local = out_dir / fname
            if local.exists():
                _upload(svc, local, fname, races_id)
                print(f'  ✓ {fname}')

    print(f'Done — {len(exported)} race cards published to Drive/PT/races/')


if __name__ == '__main__':
    main()
