"""
Seed pipeline for LOFI lineup artists.

Steps:
  1. Read 642 confirmed LOFI lineup artists from data/lofi_lineup_artists.txt
  2. Mark all matched enriched records as lofi_booked=True + lofi_lineup=True
  3. Scrape the ~11 unmatched artists via BATCH_SOURCES (Last.fm, SoundCloud, etc.)
  4. Push all 642 to Neo4j with LOFILineup label + all enriched properties
  5. Print summary; next step is: python run.py --seed

Run from repo root:
    python seed_lofi_lineup.py [--scrape-all]

    --scrape-all  Also refresh Last.fm data for all 642 (slow, ~25 min).
                  Omit for a fast run that only scrapes the 11 unmatched artists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

_ROOT         = Path(__file__).parent
_LINEUP_FILE  = _ROOT / "data" / "lofi_lineup_artists.txt"
_ENRICHED     = _ROOT / "scraper_data" / "artist_enriched.jsonl"

BATCH_SOURCES = ["Last.fm", "SoundCloud", "Discogs", "YouTube", "Mixcloud"]


def _slug(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")


# ── Load / save helpers ──────────────────────────────────────────────────────

def _load_enriched() -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not _ENRICHED.exists():
        return records
    for line in _ENRICHED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
                records[d["artist_id"]] = d
            except Exception:
                pass
    return records


def _save_enriched(records: dict[str, dict]) -> None:
    _ENRICHED.parent.mkdir(parents=True, exist_ok=True)
    with open(_ENRICHED, "w", encoding="utf-8") as f:
        for d in records.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _load_lineup() -> list[str]:
    names = []
    for line in _LINEUP_FILE.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and name.lower() != "not a headliner":
            names.append(name)
    return names


# ── Scrape helpers ───────────────────────────────────────────────────────────

def _scrape_artist(name: str) -> dict:
    """Scrape one artist across BATCH_SOURCES in parallel. Returns merged raw dict."""
    from scrapers.unified_scraper import merge_into_enriched, _SOURCE_FNS
    base = {"artist_id": _slug(name), "name": name}
    results: dict[str, dict] = {}

    def _fetch(source: str) -> tuple[str, dict | None]:
        fn = _SOURCE_FNS.get(source)
        if not fn:
            return source, None
        try:
            key = source.lower().replace(".", "").replace(" ", "_")
            data = fn(name)
            return key, data
        except Exception:
            return source.lower().replace(".", "").replace(" ", "_"), None

    with ThreadPoolExecutor(max_workers=len(BATCH_SOURCES)) as ex:
        futures = {ex.submit(_fetch, s): s for s in BATCH_SOURCES}
        for fut in as_completed(futures):
            key, data = fut.result()
            if data:
                results[key] = data

    return merge_into_enriched(base, results)


# ── Neo4j push ───────────────────────────────────────────────────────────────

def _push_to_neo4j(neo4j, aid: str, record: dict, lineup: bool = True) -> None:
    from lofi_tinder.neo4j_enrich import _career_stage, _slug as _es

    bs    = record.get("booking_stats") or {}
    gh    = record.get("growth_history") or {}
    stage = _career_stage(record)

    tags = list(dict.fromkeys(
        (record.get("lastfm_tags") or []) +
        (record.get("ra_genres") or []) +
        (record.get("spotify_genres") or [])
    ))[:6]

    listeners = gh.get("current_listeners") or record.get("spotify_followers") or 0
    growth    = gh.get("listener_growth_pct_total")

    props = {
        "name":              record.get("name", aid),
        "lofi_booked":       bool(record.get("lofi_booked")),
        "lofi_lineup":       lineup,
        "lofi_appearances":  record.get("lofi_appearance_count") or 0,
        "career_stage":      stage,
        "total_bookings":    bs.get("total") or 0,
        "bookings_12m":      bs.get("recent_12m") or 0,
        "booking_velocity":  float(bs.get("booking_velocity") or 0),
        "geo_spread":        bs.get("geo_spread") or 0,
        "nl_ratio":          float(bs.get("nl_ratio") or 0),
        "listeners":         listeners,
        "listener_growth":   float(growth) if growth is not None else 0.0,
        "momentum_score":    float(record.get("momentum_score") or 0),
        "genre_tags":        ", ".join(tags),
        "beatport_tier":     record.get("beatport_label_tier") or "",
        "pf_fans":           record.get("pf_fans") or 0,
        "spotify_followers": record.get("spotify_followers") or 0,
        "sc_followers":      record.get("sc_followers") or 0,
        "discogs_releases":  record.get("discogs_releases") or 0,
    }

    with neo4j._driver.session() as s:
        s.run("MERGE (a:Artist {artist_id: $aid}) SET a += $props", aid=aid, props=props)
        s.run("MATCH (a:Artist {artist_id: $aid}) SET a:LOFIBooked", aid=aid)
        s.run(f"MATCH (a:Artist {{artist_id: $aid}}) SET a:LOFILineup", aid=aid)
        s.run(f"MATCH (a:Artist {{artist_id: $aid}}) SET a:{stage}", aid=aid)

    similar = list(dict.fromkeys(
        (record.get("lastfm_similar") or []) +
        (record.get("spotify_related") or [])
    ))[:10]
    with neo4j._driver.session() as s:
        for sim_name in similar:
            sim_slug = _es(sim_name)
            s.run(
                """
                MERGE (a:Artist {artist_id: $aid})
                MERGE (b:Artist {artist_id: $sim_slug})
                  ON CREATE SET b.name = $sim_name
                MERGE (a)-[:SIMILAR_TO {source: 'enriched'}]->(b)
                """,
                aid=aid, sim_slug=sim_slug, sim_name=sim_name,
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed LOFI lineup artists into enriched + Neo4j")
    parser.add_argument("--scrape-all", action="store_true",
                        help="Re-scrape all 642 lineup artists (slow)")
    args = parser.parse_args()

    # ── 1. Load data
    print("Loading lineup artists...")
    lineup_names = _load_lineup()
    print(f"  {len(lineup_names)} lineup artists loaded")

    print("Loading enriched records...")
    enriched = _load_enriched()
    print(f"  {len(enriched)} existing enriched records")

    # ── 2. Classify: matched vs missing
    matched:   list[tuple[str, str]] = []  # (artist_id, name)
    missing:   list[str] = []

    for name in lineup_names:
        aid = _slug(name)
        if aid in enriched:
            matched.append((aid, name))
        else:
            missing.append(name)

    print(f"  Matched: {len(matched)} | Missing (need scrape): {len(missing)}")

    # ── 3. Mark all matched as lofi_booked + lofi_lineup
    print("Marking lineup artists as lofi_booked=True...")
    for aid, name in matched:
        enriched[aid]["lofi_booked"] = True
        enriched[aid]["lofi_lineup"] = True

    # ── 4. Scrape missing artists
    if missing:
        print(f"\nScraping {len(missing)} unmatched artists...")
        for i, name in enumerate(missing, 1):
            print(f"  [{i}/{len(missing)}] {name}")
            try:
                record = _scrape_artist(name)
                record["lofi_booked"] = True
                record["lofi_lineup"] = True
                aid = _slug(name)
                enriched[aid] = record
                print(f"    -> scraped OK")
            except Exception as e:
                print(f"    -> failed: {e}")
                # Create minimal stub so it doesn't get lost
                aid = _slug(name)
                enriched[aid] = {
                    "artist_id": aid, "name": name,
                    "lofi_booked": True, "lofi_lineup": True,
                }

    # ── 5. Optionally re-scrape all lineup artists for freshness
    if args.scrape_all:
        to_scrape = [(aid, name) for aid, name in matched]
        print(f"\nRe-scraping all {len(to_scrape)} lineup artists (--scrape-all)...")
        for i, (aid, name) in enumerate(to_scrape, 1):
            if i % 50 == 0:
                print(f"  {i}/{len(to_scrape)}")
            try:
                from scrapers.unified_scraper import merge_into_enriched, _SOURCE_FNS
                base = enriched[aid]
                raw_results: dict[str, dict] = {}

                def _fetch_src(source: str) -> tuple[str, dict | None]:
                    fn = _SOURCE_FNS.get(source)
                    key = source.lower().replace(".", "").replace(" ", "_")
                    if not fn:
                        return key, None
                    try:
                        return key, fn(name)
                    except Exception:
                        return key, None

                with ThreadPoolExecutor(max_workers=len(BATCH_SOURCES)) as ex:
                    futs = {ex.submit(_fetch_src, s): s for s in BATCH_SOURCES}
                    for fut in as_completed(futs):
                        key, data = fut.result()
                        if data:
                            raw_results[key] = data

                updated = merge_into_enriched(base, raw_results)
                updated["lofi_booked"] = True
                updated["lofi_lineup"] = True
                enriched[aid] = updated
            except Exception as e:
                print(f"  {name}: scrape error — {e}")

    # ── 6. Save updated enriched file
    print(f"\nSaving {len(enriched)} records to artist_enriched.jsonl...")
    _save_enriched(enriched)
    print("  Saved.")

    # ── 7. Push lineup artists to Neo4j
    print("\nConnecting to Neo4j...")
    from lofi_tinder.neo4j_client import get_client
    neo4j = get_client()

    if not neo4j.available:
        print("  Neo4j not available — skipping graph push.")
        print("  Check .env for NEO4J_URI / NEO4J_PASSWORD.")
    else:
        lineup_ids = {_slug(n) for n in lineup_names}
        to_push = [(aid, enriched[aid]) for aid in lineup_ids if aid in enriched]
        print(f"  Pushing {len(to_push)} artists to Neo4j...")

        ok = 0
        err = 0
        for i, (aid, record) in enumerate(to_push, 1):
            if i % 100 == 0:
                print(f"  {i}/{len(to_push)}")
            try:
                _push_to_neo4j(neo4j, aid, record, lineup=True)
                ok += 1
            except Exception as e:
                err += 1
                if err <= 5:
                    print(f"  Push error for {record.get('name', aid)}: {e}")

        print(f"  Neo4j push complete: {ok} ok, {err} errors")
        neo4j.close()

    # ── 8. Summary
    print("\nDone.")
    print(f"  Lineup artists in enriched: {sum(1 for r in enriched.values() if r.get('lofi_lineup'))}")
    print(f"  Booked artists in enriched: {sum(1 for r in enriched.values() if r.get('lofi_booked'))}")
    print()
    print("Next steps:")
    print("  python run.py --seed        # rebuild FAISS centroid from lofi_booked artists")
    print("  python run.py --candidates  # populate candidate pool")
    print("  streamlit run lofi_tinder/app.py")


if __name__ == "__main__":
    main()
