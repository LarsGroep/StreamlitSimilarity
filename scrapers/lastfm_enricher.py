"""
Last.fm enricher — fetches artist.getInfo for all artists NOT yet in LastFMSnapshot.jsonl.

Appends new records to LastFMSnapshot.jsonl in a compatible format so data_aggregator.py
picks them up automatically on next run.

Run:
    cd Testing/lofi-tinder
    python scrapers/lastfm_enricher.py
    # or via: python run.py --collect-all

Rate: ~4 req/s (0.25s sleep), free API, 1M calls/day limit.
Incremental: artists already in the snapshot are skipped.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_ROOT     = Path(__file__).parent.parent          # lofi-tinder/
_SNAPSHOT = _ROOT.parent / "ra-scraper-master" / "scraper" / "lastfm" / "LastFMSnapshot.jsonl"
_ENRICHED = _ROOT / "scraper_data" / "artist_enriched.jsonl"

API_KEY    = "5a03e4d23e2fe689339fab0a79438f20"
BASE_URL   = "https://ws.audioscrobbler.com/2.0/"
RATE_SLEEP = 0.25   # 4 req/s, safe under Last.fm's 5/s limit


def _get(method: str, artist: str, **extra) -> dict | None:
    params = {
        "method":      method,
        "artist":      artist,
        "api_key":     API_KEY,
        "format":      "json",
        "autocorrect": "1",
        **extra,
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"User-Agent": "LofiArtistScout/1.0 (lars.vandergroep@gmail.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_artist(name: str) -> dict | None:
    data = _get("artist.getInfo", name)
    if not data:
        return None
    artist = data.get("artist") or {}
    if not artist or artist.get("name") == "":
        return None

    stats    = artist.get("stats") or {}
    tag_list = (artist.get("tags") or {}).get("tag") or []
    tags     = [t["name"] for t in (tag_list if isinstance(tag_list, list) else [tag_list])]
    sim_list = (artist.get("similar") or {}).get("artist") or []
    similar  = [s["name"] for s in (sim_list if isinstance(sim_list, list) else [sim_list])]

    images  = artist.get("image") or []
    img_url = None
    for img in reversed(images):
        if img.get("#text"):
            img_url = img["#text"]
            break

    listeners = int(stats.get("listeners") or 0) or None
    playcount = int(stats.get("playcount") or 0) or None

    return {
        "name":       name,
        "query_name": name,
        "listeners":  listeners,
        "playcount":  playcount,
        "plays_per_listener": (
            round(playcount / listeners, 1) if listeners and playcount else None
        ),
        "tags":       tags[:6],
        "similar":    similar[:6],
        "image_url":  img_url,
        "is_mainstream": (listeners or 0) > 1_500_000,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    if not _ENRICHED.exists():
        print(f"ERROR: {_ENRICHED} not found — run: python run.py --enrich")
        sys.exit(1)

    all_names: list[str] = []
    for line in _ENRICHED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                all_names.append(json.loads(line)["name"])
            except Exception:
                pass

    already: set[str] = set()
    if _SNAPSHOT.exists():
        for line in _SNAPSHOT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rec  = json.loads(line)
                    name = rec.get("name") or rec.get("query_name") or ""
                    if name:
                        already.add(name.lower())
                except Exception:
                    pass

    to_fetch = [n for n in all_names if n.lower() not in already]
    print(f"Artists in enriched data:  {len(all_names)}")
    print(f"Already in Last.fm cache:  {len(already)}")
    print(f"To fetch from Last.fm API: {len(to_fetch)}  "
          f"(~{len(to_fetch) * RATE_SLEEP / 60:.1f} min)")

    if not to_fetch:
        print("Nothing to do — all artists already in Last.fm cache.")
        return

    found  = 0
    missed = 0

    with open(_SNAPSHOT, "a", encoding="utf-8") as f:
        for i, name in enumerate(to_fetch, 1):
            rec = _fetch_artist(name)
            time.sleep(RATE_SLEEP)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                found += 1
            else:
                missed += 1
            if i % 50 == 0 or i == len(to_fetch):
                pct = i / len(to_fetch) * 100
                print(f"  [{i}/{len(to_fetch)}] {pct:.0f}%  found={found}  not_on_lastfm={missed}",
                      flush=True)

    print(f"\nDone. {found} new records written to {_SNAPSHOT}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
