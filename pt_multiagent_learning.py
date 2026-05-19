"""
Multi-agent learning pipeline for PT horse racing predictions.

Phases:
  1. Judge (parallel)   — Agent 2: evaluates reasoning quality after results
  2. Extract (parallel) — Agent 3: extracts patterns from high-value verdicts only
  3. Merge              — bookkeeping: update counters in DB
  4. Curate             — Agent 4: deduplicate, generalise, condense the full DB

Usage (from PT_monitor_learning.ipynb cell 3d/3e):
    from pt_multiagent_learning import run_multi_agent_learning

    learnings_db = run_multi_agent_learning(
        enriched_races          = enriched_races,
        learnings_db            = learnings_db,
        api_key                 = ANTHROPIC_API_KEY,
        extractor_system_prompt = _LEARNING_SYSTEM_PROMPT,
        curator_system_prompt   = _LEARNING_CURATOR_PROMPT,
        judge_system_prompt     = _JUDGE_SYSTEM_PROMPT,
        today                   = TODAY,
    )
"""

import copy
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic


# ── Shared helpers ─────────────────────────────────────────────────────────────

_BUCKET_SKELETON = {
    "race_type":       [{"H": 0}, {"R": 0}, {"M": 0}, {"None": 0}],
    "going_category":  [{"VERY SLOW": 0}, {"SLOW": 0}, {"FAST": 0}, {"VERY FAST": 0}, {"PSF": 0}],
    "total_prize_eur": [{"0-10000": 0}, {"10001-20000": 0}, {"20001-30000": 0}, {"30001-55000": 0}, {">55000": 0}],
    "distance":        [{"0-1200": 0}, {"1201-1600": 0}, {"1601-2200": 0}, {"2201-2600": 0}, {">2600": 0}],
    "age_group":       [{"2yo": 0}, {"3yo": 0}, {"3yo+": 0}, {"4yo+": 0}],
    "fieldsize":       [{"0-6": 0}, {"7-12": 0}, {">12": 0}],
}


def _strip_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text


def _call_with_backoff(client: anthropic.Anthropic, **kwargs):
    """Synchronous Claude call with exponential backoff on rate-limit errors."""
    for attempt in range(4):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == 3:
                raise
            wait = 30 * (2 ** attempt)
            print(f"    ⏳ Rate limit — waiting {wait}s before retry {attempt + 1}/3 ...")
            time.sleep(wait)


# ── Agent 2: Verdict Quality Judge ────────────────────────────────────────────

def _judge_single_verdict(client: anthropic.Anthropic, race: dict, judge_prompt: str) -> dict | None:
    """
    Judge the reasoning quality of one verdict.
    Returns judgment dict, or None on failure.
    """
    label = f"{race.get('meeting', '?')} — {race.get('race', '?')}"

    # Send only the verdict + result (not full horse arrays) to keep tokens low
    payload = {k: race.get(k) for k in (
        "raceId", "meeting", "race", "going_category", "distance_group",
        "race_type", "field_size", "total_prize_eur",
        "nap", "each_way", "result",
    )}

    try:
        resp = _call_with_backoff(
            client,
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=judge_prompt,
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        judgment = json.loads(_strip_json(resp.content[0].text))
        judgment["_label"] = label
        judgment["_race_id"] = str(race.get("raceId", ""))
        return judgment
    except Exception as exc:
        print(f"  ⚠️  Judge failed [{label}]: {exc}")
        return None


# ── Agent 3: Learning Extractor ───────────────────────────────────────────────

_MIN_LEARNING_VALUE = 3   # races below this threshold are skipped


def _extract_single_race(
    client: anthropic.Anthropic,
    race: dict,
    judgment: dict | None,
    extractor_prompt: str,
    system_block: list,
) -> list:
    """
    Extract generalised learnings from one quality-judged race.
    Returns list of learning dicts (may be empty).
    """
    label = f"{race.get('meeting', '?')} — {race.get('race', '?')}"

    if judgment is None:
        print(f"  – {label}: no judgment — skipping")
        return []

    lv = judgment.get("learning_value", 0)
    if lv < _MIN_LEARNING_VALUE:
        print(f"  – {label}: learning_value={lv} < {_MIN_LEARNING_VALUE} — skipping")
        return []

    nap_oc = judgment.get("nap_outcome", "?")
    ew_oc  = judgment.get("ew_outcome", "?")

    # Build judge context injected at end of user message
    focus_hint = ""
    if "SOUND_WIN" in (nap_oc, ew_oc):
        focus_hint = "Extract patterns explaining WHY the sound reasoning correctly identified the winner/placed horse."
    elif "POOR_LOSS" in (nap_oc, ew_oc):
        focus_hint = "Identify the specific signal that was missed or misread that caused the poor prediction."
    elif "SOUND_LOSS" in (nap_oc, ew_oc):
        focus_hint = "Extract any genuine signal that correctly pointed to the selection even though it lost — or patterns from the winner that could inform future races."
    elif "LUCKY_WIN" in (nap_oc, ew_oc):
        focus_hint = "Prefer learnings about the winner's signals that the reasoning MISSED, not the lucky pick itself."

    judge_context = (
        f"\n\n── VERDICT QUALITY ASSESSMENT ─────────────────────────────────────────\n"
        f"nap_outcome:       {nap_oc}\n"
        f"ew_outcome:        {ew_oc}\n"
        f"signal_alignment:  {judgment.get('signal_alignment')}/5\n"
        f"reasoning_quality: {judgment.get('reasoning_quality')}/5\n"
        f"learning_value:    {lv}/5\n"
        f"meta_learning hint: {judgment.get('meta_learning', '')}\n"
        f"────────────────────────────────────────────────────────────────────\n"
        f"{focus_hint}\n"
        f"Only extract learnings directly supported by the race signal data above."
    )

    msgs = [{"role": "user", "content":
        json.dumps(race, indent=2, default=str) + judge_context}]

    try:
        max_tok = 4096
        resp = _call_with_backoff(
            client,
            model="claude-sonnet-4-6",
            max_tokens=max_tok,
            system=system_block,
            messages=msgs,
        )
        if resp.stop_reason == "max_tokens":
            max_tok *= 2
            print(f"  ⏳ Token limit [{label}] — retrying with max_tokens={max_tok} ...")
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=max_tok,
                system=system_block,
                messages=msgs,
            ) as stream:
                resp = stream.get_final_message()
            if resp.stop_reason == "max_tokens":
                print(f"  ⚠️  Still truncated [{label}] — skipping")
                return []

        raw = _strip_json(resp.content[0].text)
        match = re.search(r"\[\s*(?:\{[\s\S]*\})?\s*\]", raw)
        if not match:
            return []

        learnings = json.loads(match.group())
        cache_hit = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        tag = f"💾{cache_hit}t" if cache_hit else ""
        print(
            f"  ✓ {label}: {len(learnings)} learnings "
            f"[{nap_oc}/{ew_oc} lv={lv}] {tag}"
        )
        return learnings

    except Exception as exc:
        print(f"  ⚠️  Extractor failed [{label}]: {exc}")
        return []


# ── Phase 3: Merge new learnings into DB ──────────────────────────────────────

def _merge_into_db(learnings_db: list, new_learnings: list, today: str) -> list:
    """Merge extracted learnings into DB: increment counters for known IDs, append new."""
    existing_ids = {l["id"]: i for i, l in enumerate(learnings_db)}
    for nl in new_learnings:
        nid = nl.get("id", "")
        nl.setdefault("market_edge", False)
        nl.setdefault("category_counters", copy.deepcopy(_BUCKET_SKELETON))
        if nid in existing_ids:
            idx = existing_ids[nid]
            learnings_db[idx]["counter"] = learnings_db[idx].get("counter", 1) + 1
            learnings_db[idx]["last_updated"] = today
            if nl.get("market_edge"):
                learnings_db[idx]["market_edge"] = True
            learnings_db[idx].setdefault("created", today)
            print(f"  Updated: {nid} (counter={learnings_db[idx]['counter']})")
        else:
            nl["counter"] = 1
            nl["last_updated"] = today
            nl["created"] = today
            learnings_db.append(nl)
            existing_ids[nid] = len(learnings_db) - 1
            print(f"  Added:   {nid} (probation)")
    return learnings_db


# ── Agent 4: Learning Curator ─────────────────────────────────────────────────

def _curate_db(client: anthropic.Anthropic, learnings_db: list, curator_prompt: str, today: str) -> list:
    """Curate the full DB: deduplicate, generalise, condense, remove noise."""
    if not learnings_db:
        return learnings_db

    curate_input = (
        f"Today is {today}. Curate this learning database:\n"
        + json.dumps(learnings_db, separators=(",", ":"), default=str)
    )
    print(f"Curator input: ~{len(curate_input) // 4} tokens")

    def _run_stream(max_tok):
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=max_tok,
            system=curator_prompt,
            messages=[{"role": "user", "content": curate_input}],
        ) as s:
            return s.get_final_message()

    resp = _run_stream(65536)
    if resp.stop_reason == "max_tokens":
        print("⚠️  Curator truncated at 65536 — retrying with 131072 ...")
        resp = _run_stream(131072)
    if resp.stop_reason == "max_tokens":
        print("⚠️  Curator still truncated — keeping uncurated DB")
        return learnings_db

    raw = _strip_json(resp.content[0].text)
    match = re.search(r"\[\s*(?:\{[\s\S]*\})?\s*\]", raw)
    if not match:
        print("Curator returned no valid JSON — keeping uncurated DB")
        return learnings_db

    try:
        curated = json.loads(match.group())
        if not curated:
            print("Curator returned empty list — keeping current DB")
            return learnings_db
        prev = len(learnings_db)
        for l in curated:
            l.setdefault("market_edge", False)
        print(
            f"Curated: {prev} → {len(curated)} learnings "
            f"(in={resp.usage.input_tokens} / out={resp.usage.output_tokens} tokens)"
        )
        return curated
    except json.JSONDecodeError as exc:
        print(f"Curator JSON parse error: {exc} — keeping uncurated DB")
        return learnings_db


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_multi_agent_learning(
    enriched_races: list,
    learnings_db: list,
    api_key: str,
    extractor_system_prompt: str,
    curator_system_prompt: str,
    judge_system_prompt: str,
    today: str,
    max_workers: int = 5,
) -> list:
    """
    Run the full multi-agent learning pipeline.

    Phase 1 (parallel): Judge all race verdicts for reasoning quality.
    Phase 2 (parallel): Extract learnings from high-value verdicts only.
    Phase 3:            Merge new learnings into the DB (counter bookkeeping).
    Phase 4:            Curate the full DB (deduplicate / generalise / condense).

    Returns the updated learnings_db list.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Cached system block for extractor — reused across parallel threads;
    # Anthropic's 5-min ephemeral cache means the first write benefits all
    # subsequent calls that start within that window.
    extractor_system_block = [
        {"type": "text", "text": extractor_system_prompt,
         "cache_control": {"type": "ephemeral"}}
    ]

    # ── Phase 1: Judge all verdicts in parallel ───────────────────────────────
    print(f"\n── Phase 1: Judging {len(enriched_races)} verdicts (parallel, {max_workers} workers) ──")
    judgments: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(_judge_single_verdict, client, race, judge_system_prompt):
                str(race.get("raceId", i))
            for i, race in enumerate(enriched_races)
        }
        for future in as_completed(future_to_id):
            race_id = future_to_id[future]
            try:
                judgments[race_id] = future.result()
            except Exception as exc:
                print(f"  ⚠️  Judge thread error raceId={race_id}: {exc}")
                judgments[race_id] = None

    # Summarise
    outcomes = [j.get("nap_outcome", "?") for j in judgments.values() if j]
    print(f"Outcome breakdown: {dict(Counter(outcomes))}")
    high_value = sum(1 for j in judgments.values() if j and j.get("learning_value", 0) >= _MIN_LEARNING_VALUE)
    skipped = len(enriched_races) - high_value
    print(f"High learning-value (lv≥{_MIN_LEARNING_VALUE}): {high_value}  |  Skipped: {skipped}")

    # ── Phase 2: Extract learnings in parallel ────────────────────────────────
    print(f"\n── Phase 2: Extracting learnings (parallel, {max_workers} workers) ─────────────")
    new_learnings: list = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_label = {
            pool.submit(
                _extract_single_race,
                client,
                race,
                judgments.get(str(race.get("raceId", i))),
                extractor_system_prompt,
                extractor_system_block,
            ): f"{race.get('meeting','?')} — {race.get('race','?')}"
            for i, race in enumerate(enriched_races)
        }
        for future in as_completed(future_to_label):
            try:
                result = future.result()
                new_learnings.extend(result)
            except Exception as exc:
                print(f"  ⚠️  Extractor thread error: {exc}")

    print(f"\nTotal new learnings extracted: {len(new_learnings)}")
    for l in new_learnings:
        print(f"  [{l.get('direction','?')}] {l.get('id','?')}: {l.get('learning','')[:80]}")

    # ── Phase 3: Merge into DB ────────────────────────────────────────────────
    print(f"\n── Phase 3: Merging into DB ({len(learnings_db)} existing) ────────────────────")
    learnings_db = _merge_into_db(learnings_db, new_learnings, today)
    print(f"DB size after merge: {len(learnings_db)}")

    # ── Phase 4: Curate ───────────────────────────────────────────────────────
    print(f"\n── Phase 4: Curating learning database ──────────────────────────────────────")
    learnings_db = _curate_db(client, learnings_db, curator_system_prompt, today)

    return learnings_db
