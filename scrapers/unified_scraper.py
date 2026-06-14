"""
Unified per-artist scraper — all data sources in one place.

Fetches Last.fm, Spotify, SoundCloud, Discogs, YouTube, and Mixcloud data
for a given list of artists.  Designed for the Streamlit scrape phase: each
source runs in sequence so a per-source progress bar can fill cleanly.

Partyflock, Resident Advisor, and Beatport are NOT included here — they rely
on Scrapy batch spiders and are served from the pre-built artist_enriched.jsonl.

Usage (from app):
    from scrapers.unified_scraper import SOURCES, scrape_batch, merge_into_enriched

Usage (standalone):
    python scrapers/unified_scraper.py "Chris Stussy" "Josh Baker"
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# All sources displayed in the UI — order determines progress bar order
SOURCES = ["Last.fm", "Spotify", "SoundCloud", "Discogs", "YouTube", "Mixcloud"]

# Progress callback: (source_name, done_count, total_count, current_artist_name)
ProgressCB = Callable[[str, int, int, str], None]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _http_get(url: str, headers: dict | None = None, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Last.fm
# ─────────────────────────────────────────────────────────────────────────────

_LFM_KEY      = "5a03e4d23e2fe689339fab0a79438f20"
_LFM_BASE     = "https://ws.audioscrobbler.com/2.0/"
_LFM_SLEEP    = 0.25


def _lfm_get(method: str, artist: str) -> dict:
    params = {
        "method": method, "artist": artist,
        "api_key": _LFM_KEY, "format": "json", "autocorrect": "1",
    }
    url = _LFM_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"User-Agent": "LofiArtistScout/1.0 (lars.vandergroep@gmail.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8")) or {}
    except Exception:
        return {}


def _scrape_lastfm(name: str) -> dict | None:
    time.sleep(_LFM_SLEEP)
    data   = _lfm_get("artist.getInfo", name)
    artist = data.get("artist") or {}
    if not artist or not artist.get("name"):
        return None
    stats    = artist.get("stats") or {}
    tag_list = (artist.get("tags") or {}).get("tag") or []
    tags     = [t["name"] for t in (tag_list if isinstance(tag_list, list) else [tag_list])]
    sim_list = (artist.get("similar") or {}).get("artist") or []
    similar  = [s["name"] for s in (sim_list if isinstance(sim_list, list) else [sim_list])]
    images   = artist.get("image") or []
    img_url  = next((i["#text"] for i in reversed(images) if i.get("#text")), None)
    listeners = int(stats.get("listeners") or 0) or None
    playcount = int(stats.get("playcount") or 0) or None
    return {
        "name":      name,
        "listeners": listeners,
        "playcount": playcount,
        "tags":      tags[:6],
        "similar":   similar[:6],
        "image_url": img_url,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spotify
# ─────────────────────────────────────────────────────────────────────────────

_SP_SLEEP = 0.4
_sp_token: dict = {"token": None, "expires_at": 0.0}


def _sp_refresh() -> str:
    cid = os.environ.get("SPOTIFY_CLIENT_ID", "")
    sec = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not cid or not sec:
        raise RuntimeError("Spotify credentials not set")
    creds = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=b"grant_type=client_credentials",
        headers={"Authorization": f"Basic {creds}",
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["access_token"]


def _sp_token_get() -> str:
    if time.time() > _sp_token["expires_at"] - 60:
        _sp_token["token"]      = _sp_refresh()
        _sp_token["expires_at"] = time.time() + 3600
    return _sp_token["token"]


def _sp_get(url: str, retries: int = 3) -> dict:
    import re
    tok = _sp_token_get()
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "2"))
                time.sleep(retry_after + 1)
            elif e.code == 401:
                _sp_token["expires_at"] = 0
                tok = _sp_token_get()
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            else:
                break
        except Exception:
            break
    return {}


def _scrape_spotify(name: str) -> dict | None:
    time.sleep(_SP_SLEEP)
    import re
    clean = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()
    best  = None
    for q in dict.fromkeys([clean, name]):
        data  = _sp_get(f"https://api.spotify.com/v1/search?q={urllib.parse.quote(q)}&type=artist&limit=5")
        items = (data.get("artists") or {}).get("items") or []
        if items:
            nl = name.lower(); cl = clean.lower()
            best = next((i for i in items if i["name"].lower() in (nl, cl)), items[0])
            break
    if not best:
        return None
    try:
        full = _sp_get(f"https://api.spotify.com/v1/artists/{best['id']}")
        if full.get("id"):
            best = full
    except Exception:
        pass
    images  = best.get("images") or []
    img_url = images[min(1, len(images)-1)].get("url") if images else None
    related: list[str] = []
    try:
        rel  = _sp_get(f"https://api.spotify.com/v1/artists/{best['id']}/related-artists")
        related = [a["name"] for a in (rel.get("artists") or [])[:8]]
    except Exception:
        pass
    followers = (best.get("followers") or {}).get("total")
    return {
        "spotify_id":   best["id"],
        "spotify_url":  (best.get("external_urls") or {}).get("spotify"),
        "followers":    followers,
        "popularity":   best.get("popularity"),
        "genres":       best.get("genres") or [],
        "related":      related,
        "image_url":    img_url,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SoundCloud  (extracts client_id from homepage JS — no auth needed)
# ─────────────────────────────────────────────────────────────────────────────

_SC_SLEEP      = 0.4
_SC_SEARCH_URL = "https://api-v2.soundcloud.com/search/users"
_sc_client_id: list[str] = []   # mutable singleton — populated on first call


def _sc_get_client_id() -> str:
    if _sc_client_id:
        return _sc_client_id[0]
    import re as _re
    req = urllib.request.Request(
        "https://soundcloud.com",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    for script_url in _re.findall(r'"(https://a-v2\.sndcdn\.com/assets/[^"]+\.js)"', html):
        try:
            req2 = urllib.request.Request(script_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=15) as r:
                js = r.read().decode("utf-8", errors="ignore")
            m = _re.search(r'client_id:"([a-zA-Z0-9]{20,})"', js)
            if m:
                _sc_client_id.append(m.group(1))
                return _sc_client_id[0]
        except Exception:
            continue
    return ""


def _scrape_soundcloud(name: str) -> dict | None:
    import re as _re
    time.sleep(_SC_SLEEP)
    client_id = _sc_get_client_id()
    if not client_id:
        return None
    clean = _re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()
    best  = None
    for query in dict.fromkeys([clean, name]):
        params = urllib.parse.urlencode({"q": query, "limit": "5", "client_id": client_id})
        try:
            req = urllib.request.Request(
                f"{_SC_SEARCH_URL}?{params}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                items = json.loads(r.read()).get("collection") or []
        except Exception:
            continue
        if not items:
            continue
        nl = name.lower(); cl = clean.lower()
        exact = next(
            (i for i in items if (i.get("full_name") or i.get("username") or "").lower()
             in (nl, cl)),
            None,
        )
        if exact:
            best = exact; break
        top = items[0]
        if (top.get("followers_count") or 0) >= 500 or (top.get("track_count") or 0) >= 5:
            best = top; break
    if not best:
        return None
    return {
        "sc_id":           best.get("id"),
        "sc_username":     best.get("permalink"),
        "sc_url":          best.get("permalink_url"),
        "followers":       best.get("followers_count"),
        "tracks":          best.get("track_count"),
        "verified":        best.get("verified", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Discogs
# ─────────────────────────────────────────────────────────────────────────────

_DG_SLEEP = 1.1
_DG_UA    = "LOFIArtistIntelligence/1.0 (contact@lofiamsterdam.nl)"


def _dg_get(url: str) -> dict:
    key = os.environ.get("DISCOGS_KEY", "")
    sec = os.environ.get("DISCOGS_SECRET", "")
    sep  = "&" if "?" in url else "?"
    full = f"{url}{sep}key={key}&secret={sec}" if key else url
    req  = urllib.request.Request(full, headers={"User-Agent": _DG_UA})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def _scrape_discogs(name: str) -> dict | None:
    time.sleep(_DG_SLEEP)
    q    = urllib.parse.quote(name)
    data = _dg_get(f"https://api.discogs.com/database/search?q={q}&type=artist&per_page=5")
    results = data.get("results") or []
    if not results:
        return None
    nl   = name.lower()
    best = next((r for r in results if r.get("title", "").lower() == nl), results[0])
    aid  = best.get("id")
    if not aid:
        return None
    # Releases
    time.sleep(_DG_SLEEP)
    rdata    = _dg_get(f"https://api.discogs.com/artists/{aid}/releases?per_page=100&sort=year&sort_order=asc")
    releases = rdata.get("releases") or []
    labels: list[str] = []
    styles: list[str] = []
    years:  list[int] = []
    for rel in releases:
        if rel.get("label"):
            labels.append(rel["label"])
        for s in (rel.get("style") or []):
            styles.append(s)
        try:
            y = int(rel.get("year") or 0)
            if 1980 <= y <= 2030:
                years.append(y)
        except (ValueError, TypeError):
            pass
    total = (rdata.get("pagination") or {}).get("items") or len(releases)
    return {
        "discogs_id":   aid,
        "discogs_url":  f"https://www.discogs.com/artist/{aid}",
        "release_count": total,
        "top_labels":   [l for l, _ in Counter(labels).most_common(5)],
        "styles":       [s for s, _ in Counter(styles).most_common(5)],
        "first_year":   min(years) if years else None,
        "latest_year":  max(years) if years else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# YouTube
# ─────────────────────────────────────────────────────────────────────────────

_YT_SLEEP = 0.5
_YT_BASE  = "https://www.googleapis.com/youtube/v3"

_MILESTONE_CHANNELS = {
    "Boiler Room": "boiler_room",
    "RA Exchange":  "ra_exchange",
}


def _scrape_youtube(name: str) -> dict | None:
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        return None
    time.sleep(_YT_SLEEP)
    q    = urllib.parse.quote(f"{name} official")
    data = _http_get(f"{_YT_BASE}/search?part=snippet&q={q}&type=channel&maxResults=3&key={key}")
    items = data.get("items") or []
    if not items:
        return None
    nl   = name.lower()
    best = next(
        (i for i in items if nl in i["snippet"]["channelTitle"].lower()),
        items[0],
    )
    channel_id = best["id"]["channelId"]
    time.sleep(_YT_SLEEP)
    sdata  = _http_get(f"{_YT_BASE}/channels?part=statistics&id={channel_id}&key={key}")
    sitems = sdata.get("items") or []
    stats  = sitems[0].get("statistics") or {} if sitems else {}
    result = {
        "yt_channel_id":  channel_id,
        "yt_channel":     best["snippet"]["channelTitle"],
        "yt_subscribers": int(stats.get("subscriberCount") or 0) or None,
        "yt_views":       int(stats.get("viewCount")       or 0) or None,
        "yt_videos":      int(stats.get("videoCount")      or 0) or None,
    }
    for ch_name, key_name in _MILESTONE_CHANNELS.items():
        time.sleep(_YT_SLEEP)
        q2   = urllib.parse.quote(f"{name} {ch_name}")
        vdata = _http_get(
            f"{_YT_BASE}/search?part=snippet&q={q2}&type=video&maxResults=3&key={key}"
        )
        vitems = vdata.get("items") or []
        result[key_name] = any(
            nl in (i["snippet"].get("title") or "").lower()
            and ch_name.lower() in (i["snippet"].get("channelTitle") or "").lower()
            for i in vitems
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Mixcloud
# ─────────────────────────────────────────────────────────────────────────────

_MC_SLEEP = 0.3
_MC_BASE  = "https://api.mixcloud.com"


def _mc_get(url: str) -> dict:
    cid  = os.environ.get("MIXCLOUD_CLIENT_ID", "")
    full = f"{url}{'&' if '?' in url else '?'}client_id={cid}" if cid else url
    req  = urllib.request.Request(full, headers={"User-Agent": "LOFIArtistIntelligence/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def _scrape_mixcloud(name: str) -> dict | None:
    time.sleep(_MC_SLEEP)
    q    = urllib.parse.quote(name)
    data = _mc_get(f"{_MC_BASE}/search/?q={q}&type=user&limit=5")
    items = data.get("data") or []
    if not items:
        return None
    nl   = name.lower()
    best = next((i for i in items if i.get("name", "").lower() == nl), items[0])
    mc_key  = best.get("key") or ""
    profile: dict = {}
    if mc_key:
        time.sleep(_MC_SLEEP)
        profile = _mc_get(f"{_MC_BASE}{mc_key}")
    username = profile.get("username") or best.get("username") or ""
    return {
        "mc_username":    username,
        "mc_url":         best.get("url") or (f"https://www.mixcloud.com{mc_key}" if mc_key else None),
        "mc_followers":   profile.get("follower_count") or best.get("follower_count"),
        "mc_listen_count":profile.get("listen_count"),
        "mc_track_count": profile.get("track_count"),
        "mc_city":        (profile.get("city") or "").strip() or None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_FNS = {
    "Last.fm":    _scrape_lastfm,
    "Spotify":    _scrape_spotify,
    "SoundCloud": _scrape_soundcloud,
    "Discogs":    _scrape_discogs,
    "YouTube":    _scrape_youtube,
    "Mixcloud":   _scrape_mixcloud,
}


def scrape_batch(
    names: list[str],
    progress_cb: ProgressCB | None = None,
    sources: list[str] | None = None,
) -> dict[str, dict]:
    """
    Scrape `names` across all sources, source-by-source.

    progress_cb(source_name, done_count, total_count, current_artist_name)
    Returns {name: {source_key: raw_dict | None}}
    """
    active = sources or SOURCES
    n      = len(names)
    results: dict[str, dict] = {name: {} for name in names}

    for source in active:
        fn = _SOURCE_FNS.get(source)
        if not fn:
            continue
        for i, name in enumerate(names):
            try:
                data = fn(name)
                if data:
                    key = source.lower().replace(".", "").replace(" ", "_")
                    results[name][key] = data
            except Exception:
                pass
            if progress_cb:
                progress_cb(source, i + 1, n, name)

    return results


def merge_into_enriched(enriched: dict, raw: dict) -> dict:
    """
    Merge fresh scrape output (from scrape_batch) into an existing enriched dict.
    Only overwrites a field if the new value is non-None.
    """
    out = dict(enriched)

    lfm = raw.get("lastfm") or {}
    if lfm:
        gh = out.setdefault("growth_history", {})
        if lfm.get("listeners") is not None:
            gh["current_listeners"] = lfm["listeners"]
        if lfm.get("playcount") is not None:
            gh["current_playcount"] = lfm["playcount"]
        if lfm.get("tags"):
            out["lastfm_tags"] = lfm["tags"]
        if lfm.get("similar"):
            out["lastfm_similar"] = lfm["similar"]
        if lfm.get("image_url") and not out.get("image_url"):
            out["image_url"] = lfm["image_url"]

    sp = raw.get("spotify") or {}
    if sp:
        for k, v in [
            ("spotify_id",       sp.get("spotify_id")),
            ("spotify_url",      sp.get("spotify_url")),
            ("spotify_followers",sp.get("followers")),
            ("spotify_popularity", sp.get("popularity")),
        ]:
            if v is not None:
                out[k] = v
        if sp.get("genres"):
            out["spotify_genres"] = sp["genres"]
        if sp.get("related"):
            out["spotify_related"] = sp["related"]
        if sp.get("image_url") and not out.get("image_url"):
            out["image_url"] = sp["image_url"]

    sc = raw.get("soundcloud") or {}
    if sc:
        if sc.get("sc_id") is not None:
            out["sc_id"]  = sc["sc_id"]
        if sc.get("sc_url"):
            out["sc_url"] = sc["sc_url"]
        if sc.get("followers") is not None:
            # Append a snapshot so the growth chart builds over time
            snap = {
                "date":      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "followers": sc["followers"],
            }
            snaps = list(out.get("sc_snapshots") or [])
            if not snaps or snaps[-1].get("date") != snap["date"]:
                snaps.append(snap)
            out["sc_snapshots"] = snaps
            out["sc_followers"] = sc["followers"]
        if sc.get("tracks") is not None:
            out["sc_tracks"] = sc["tracks"]

    dg = raw.get("discogs") or {}
    if dg:
        for k, v in [
            ("discogs_id",       dg.get("discogs_id")),
            ("discogs_url",      dg.get("discogs_url")),
            ("discogs_releases", dg.get("release_count")),
            ("discogs_first_year", dg.get("first_year")),
        ]:
            if v is not None:
                out[k] = v
        if dg.get("top_labels"):
            out["discogs_labels"] = dg["top_labels"]
        if dg.get("styles"):
            out["discogs_styles"] = dg["styles"]

    yt = raw.get("youtube") or {}
    if yt:
        for k, v in [
            ("yt_channel_id",  yt.get("yt_channel_id")),
            ("yt_subscribers", yt.get("yt_subscribers")),
            ("yt_views",       yt.get("yt_views")),
        ]:
            if v is not None:
                out[k] = v
        out["yt_boiler_room"] = yt.get("boiler_room", False)
        out["yt_ra_exchange"] = yt.get("ra_exchange", False)

    mc = raw.get("mixcloud") or {}
    if mc:
        for k, v in [
            ("mc_followers",   mc.get("mc_followers")),
            ("mc_listen_count",mc.get("mc_listen_count")),
            ("mc_track_count", mc.get("mc_track_count")),
        ]:
            if v is not None:
                out[k] = v

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    names = sys.argv[1:] or ["Chris Stussy", "Josh Baker"]
    print(f"Scraping {len(names)} artist(s) across {len(SOURCES)} sources\n")

    def _cb(source: str, done: int, total: int, name: str) -> None:
        pct = done / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  {source:10s}  [{bar}] {done:3d}/{total}  {name[:30]:30s}", end="", flush=True)
        if done == total:
            print()

    results = scrape_batch(names, progress_cb=_cb)
    for name, data in results.items():
        print(f"\n{name}:")
        for src, d in data.items():
            keys = list(d.keys())[:4]
            print(f"  {src}: {keys}")
