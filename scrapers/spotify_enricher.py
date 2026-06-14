"""
Spotify artist enricher — batch-fetches Spotify data for all artists in artist_enriched.jsonl.

Adds per-artist: image URL, Spotify ID/URL, followers, popularity, genres, related artists.
Output: scraper_data/spotify_artists.jsonl  (one JSON record per line, keyed by artist name)

Run:
    cd Testing/lofi-tinder
    python scrapers/spotify_enricher.py
    # or via: python run.py --collect-all

Rate limit: ~2 workers × 0.4s sleep, well within Spotify's limit.
Incremental: already-fetched artists are skipped on re-run.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_ROOT     = Path(__file__).parent.parent          # lofi-tinder/
load_dotenv(_ROOT / ".env")

_ENRICHED = _ROOT / "scraper_data" / "artist_enriched.jsonl"
_OUT      = _ROOT / "scraper_data" / "spotify_artists.jsonl"

_MAX_WORKERS = 2
_RATE_SLEEP  = 0.4    # per-worker delay — ~5 req/s total, safely under Spotify's limit

_tok_lock  = threading.Lock()
_tok_state: dict = {"token": None, "expires_at": 0.0}


def _refresh_token() -> str:
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in .env")
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    return data["access_token"]


def _get_token() -> str:
    with _tok_lock:
        if time.time() > _tok_state["expires_at"] - 60:
            _tok_state["token"]      = _refresh_token()
            _tok_state["expires_at"] = time.time() + 3600
        return _tok_state["token"]


def _api_get(url: str, _retries: int = 3) -> dict:
    import urllib.error
    tok = _get_token()
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    for attempt in range(_retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "2"))
                time.sleep(retry_after + 1)
                tok = _get_token()
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            elif e.code == 401:
                with _tok_lock:
                    _tok_state["expires_at"] = 0
                tok = _get_token()
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            else:
                raise
    return {}


def _clean_name(name: str) -> str:
    import re
    return re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()


def _search_artist(name: str) -> dict | None:
    clean = _clean_name(name)
    for query in dict.fromkeys([clean, name]):
        q   = urllib.parse.quote(query)
        url = f"https://api.spotify.com/v1/search?q={q}&type=artist&limit=5"
        try:
            data  = _api_get(url)
            items = (data.get("artists") or {}).get("items") or []
        except Exception:
            return None
        if not items:
            continue
        name_lower  = name.lower()
        clean_lower = clean.lower()
        best = next(
            (i for i in items if i["name"].lower() in (name_lower, clean_lower)),
            items[0],
        )
        return best
    return None


def _enrich(name: str) -> dict | None:
    time.sleep(_RATE_SLEEP)
    best = _search_artist(name)
    if not best:
        return None

    images  = best.get("images") or []
    img_url = None
    img_sm  = None
    if images:
        img_url = images[min(1, len(images) - 1)].get("url")
        img_sm  = images[-1].get("url")

    related: list[str] = []
    try:
        rel_data = _api_get(f"https://api.spotify.com/v1/artists/{best['id']}/related-artists")
        related  = [a["name"] for a in (rel_data.get("artists") or [])[:8]]
    except Exception:
        pass

    return {
        "name":           name,
        "spotify_id":     best["id"],
        "spotify_name":   best["name"],
        "spotify_url":    (best.get("external_urls") or {}).get("spotify"),
        "followers":      (best.get("followers") or {}).get("total"),
        "popularity":     best.get("popularity"),
        "genres":         best.get("genres") or [],
        "image_url":      img_url,
        "image_url_thumb":img_sm,
        "related_artists":related,
        "scraped_at":     datetime.now(timezone.utc).isoformat(),
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
    print(f"Artists to enrich: {len(all_names)}")

    cached: dict[str, dict] = {}
    if _OUT.exists():
        for line in _OUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    cached[rec["name"]] = rec
                except Exception:
                    pass
    print(f"Already cached: {len(cached)}")

    to_fetch = [n for n in all_names if n not in cached]
    print(f"To fetch: {len(to_fetch)}  (~{len(to_fetch) * _RATE_SLEEP / _MAX_WORKERS:.0f}s estimated)")

    if not to_fetch:
        print("Nothing to do — all artists already cached.")
        return

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    lock   = threading.Lock()
    done   = 0
    found  = 0
    errors = 0

    with open(_OUT, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_enrich, name): name for name in to_fetch}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = None
                    with lock:
                        errors += 1
                with lock:
                    done += 1
                    if result:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        found += 1
                    if done % 100 == 0 or done == len(to_fetch):
                        pct = done / len(to_fetch) * 100
                        print(f"  [{done}/{len(to_fetch)}] {pct:.0f}% — found {found}, errors {errors}",
                              flush=True)

    print(f"\nDone. {found} new records fetched. Total cached: {len(cached) + found}")
    print(f"Output: {_OUT}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
