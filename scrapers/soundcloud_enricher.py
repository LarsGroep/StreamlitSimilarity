"""
SoundCloud artist enricher — batch-fetches follower/track data for all artists.

Each run APPENDS a timestamped snapshot, building a time series over repeated runs.
Underground artists often have SoundCloud as their primary platform before Spotify.

Output: scraper_data/soundcloud_artists.jsonl  (one JSON record per line per run)

Run:
    cd Testing/lofi-tinder
    python scrapers/soundcloud_enricher.py
    # or via: python run.py --collect-all

Incremental per-run: fetches all artists, appends new snapshots (no skip).
Re-running weekly builds growth history visible in the Tinder UI.
"""

from __future__ import annotations

import json
import re
import sys
import time
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_ROOT     = Path(__file__).parent.parent          # lofi-tinder/
_ENRICHED = _ROOT / "scraper_data" / "artist_enriched.jsonl"
_OUT      = _ROOT / "scraper_data" / "soundcloud_artists.jsonl"

_MAX_WORKERS  = 3
_RATE_SLEEP   = 0.35   # per-worker delay

_SC_HOMEPAGE   = "https://soundcloud.com"
_SC_SEARCH_URL = "https://api-v2.soundcloud.com/search/users"

_client_id_lock = threading.Lock()
_client_id_cache: dict = {"id": None}


def _get_client_id() -> str:
    with _client_id_lock:
        if _client_id_cache["id"]:
            return _client_id_cache["id"]
        req = urllib.request.Request(
            _SC_HOMEPAGE,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="ignore")
        script_urls = re.findall(
            r'"(https://a-v2\.sndcdn\.com/assets/[^"]+\.js)"', html
        )
        for url in script_urls:
            try:
                req2 = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req2, timeout=15) as r:
                    js = r.read().decode("utf-8", errors="ignore")
                m = re.search(r'client_id:"([a-zA-Z0-9]{20,})"', js)
                if m:
                    _client_id_cache["id"] = m.group(1)
                    return _client_id_cache["id"]
            except Exception:
                continue
        raise RuntimeError("Could not extract SoundCloud client_id from homepage JS")


def _clean_name(name: str) -> str:
    return re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()


def _search_artist(name: str, client_id: str) -> dict | None:
    clean = _clean_name(name)
    for query in dict.fromkeys([clean, name]):
        params = urllib.parse.urlencode({"q": query, "limit": "5", "client_id": client_id})
        url = f"{_SC_SEARCH_URL}?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                items = json.loads(r.read()).get("collection") or []
        except Exception:
            return None
        if not items:
            continue
        name_lower  = name.lower()
        clean_lower = clean.lower()
        exact = next(
            (i for i in items if (i.get("full_name") or i.get("username") or "").lower()
             in (name_lower, clean_lower)),
            None,
        )
        if exact:
            return exact
        top = items[0]
        if (top.get("followers_count") or 0) >= 500 or (top.get("track_count") or 0) >= 5:
            return top
    return None


def _enrich(name: str) -> dict | None:
    time.sleep(_RATE_SLEEP)
    try:
        client_id = _get_client_id()
        best = _search_artist(name, client_id)
    except Exception:
        return None
    if not best:
        return None

    return {
        "name":            name,
        "sc_id":           best.get("id"),
        "sc_username":     best.get("permalink"),
        "sc_display_name": best.get("full_name") or best.get("username"),
        "sc_url":          best.get("permalink_url"),
        "followers":       best.get("followers_count"),
        "following":       best.get("followings_count"),
        "tracks":          best.get("track_count"),
        "reposts":         best.get("reposts_count"),
        "likes":           best.get("likes_count"),
        "description":     (best.get("description") or "")[:200] or None,
        "verified":        best.get("verified", False),
        "scraped_at":      datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    if not _ENRICHED.exists():
        print(f"ERROR: {_ENRICHED} not found — run: python run.py --enrich")
        return

    all_names: list[str] = []
    for line in _ENRICHED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                all_names.append(json.loads(line)["name"])
            except Exception:
                pass
    print(f"Artists to fetch: {len(all_names)}")
    print(f"Estimated time: ~{len(all_names) * _RATE_SLEEP / _MAX_WORKERS / 60:.0f} min")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    already_today: set[str] = set()
    if _OUT.exists():
        for line in _OUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if (rec.get("scraped_at") or "")[:10] == today:
                        already_today.add(rec["name"])
                except Exception:
                    pass

    to_fetch = [n for n in all_names if n not in already_today]
    print(f"Skipping {len(already_today)} already fetched today. Fetching {len(to_fetch)}.")

    if not to_fetch:
        print("Already ran today. Re-run tomorrow to build time-series history.")
        return

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    lock   = threading.Lock()
    done   = found = errors = 0

    with open(_OUT, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_enrich, name): name for name in to_fetch}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except Exception:
                    result = None
                with lock:
                    done += 1
                    if result:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        found += 1
                    else:
                        errors += 1
                    if done % 200 == 0 or done == len(to_fetch):
                        pct = done / len(to_fetch) * 100
                        print(f"  [{done}/{len(to_fetch)}] {pct:.0f}%  found={found}  not_on_sc={errors}",
                              flush=True)

    print(f"\nDone. {found} new records written -> {_OUT}")
    print("Run: python run.py --enrich  to incorporate SoundCloud data.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
