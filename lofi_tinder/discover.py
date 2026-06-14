"""
Discovery loop — finds new artist candidates from Last.fm similar-artist chains.

Called after each batch of 20 swipes. Uses YES'd artists' stored similar-artist
lists to surface new candidates, fetches their Last.fm data, generates profiles,
and adds them to the candidate pool.

Usage (from app.py):
    from lofi_tinder.discover import discover_new_batch
    new_ids = discover_new_batch(yes_names, swiped_ids, profiles)
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
import re
from datetime import datetime, timezone
from pathlib import Path

_ROOT              = Path(__file__).parent.parent
_RA_SCRAPER        = _ROOT.parent / "ra-scraper-master" / "scraper"
_PROFILES_FILE     = _ROOT / "profiles" / "artist_profiles.jsonl"
_CANDIDATES_FILE   = _ROOT / "data" / "candidates.jsonl"
_ENRICHED_FILE     = _ROOT / "scraper_data" / "artist_enriched.jsonl"
_SEED_HISTORY_FILE = _ROOT / "data" / "discovery_seed_history.jsonl"

sys.path.insert(0, str(_RA_SCRAPER / "lastfm"))


def _slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def _load_seed_history() -> dict[str, int]:
    """Return {name_lower: times_used_as_seed}. More uses = less preferred next round."""
    counts: dict[str, int] = {}
    if not _SEED_HISTORY_FILE.exists():
        return counts
    for line in _SEED_HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                name = (json.loads(line).get("name") or "").lower()
                if name:
                    counts[name] = counts.get(name, 0) + 1
            except Exception:
                pass
    return counts


def _record_seeds_used(names: list[str]) -> None:
    """Append used seed names to the history file."""
    _SEED_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(_SEED_HISTORY_FILE, "a", encoding="utf-8") as f:
        for name in names:
            f.write(json.dumps({"name": name, "used_at": now}, ensure_ascii=False) + "\n")


def _select_seeds(yes_names: list[str], seed_history: dict[str, int], max_seeds: int = 8) -> list[str]:
    """
    Pick up to max_seeds YES artists to use as Last.fm seed queries.

    Prioritises artists used least often as seeds — this prevents the discovery
    loop from always expanding the same corner of Last.fm's similarity graph and
    collapsing into a self-reinforcing feedback loop.

    Within the same use-count tier, artists from the EDGE of the YES pool
    (those added most recently) are preferred to inject recency without losing variety.
    """
    # Sort ascending by times-used, then by reverse position (most recent first)
    indexed = [(i, name) for i, name in enumerate(yes_names)]
    indexed.sort(key=lambda x: (seed_history.get(x[1].lower(), 0), -x[0]))
    return [name for _, name in indexed[:max_seeds]]


def _load_existing_slugs() -> set[str]:
    slugs: set[str] = set()
    if not _PROFILES_FILE.exists():
        return slugs
    for line in _PROFILES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                slugs.add(json.loads(line)["artist_id"])
            except Exception:
                pass
    return slugs


def _load_enriched_map() -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not _ENRICHED_FILE.exists():
        return result
    for line in _ENRICHED_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                result[data["artist_id"]] = data
                for old_id in data.get("old_artist_ids") or []:
                    result[old_id] = data
            except Exception:
                pass
    return result


def _collect_similar_names(yes_names: list[str]) -> list[str]:
    """Call Last.fm artist.getSimilar live — returns 50 candidates per YES artist.
    Much broader than stored lastfm_similar (avg 4.2 per record) and fresh each call."""
    from lastfm_scraper import fetch_similar_artists
    seen: dict[str, None] = {}
    for name in yes_names:
        for n in fetch_similar_artists(name, limit=50):
            seen.setdefault(n)
        time.sleep(0.25)
    return list(seen.keys())


def _extract_filter_params(yes_names: list[str], enriched_map: dict[str, dict]) -> dict:
    """
    Translate the LOFI Feel Matrix into concrete discovery filter parameters.

    The Feel Matrix (centroid of YES'd artist embeddings) is high-dimensional and
    abstract. This function extracts human-readable parameters from the structured
    enriched data of YES'd artists:
      - top_tags: genre tags most common in YES'd artists → used for tag.getTopArtists
      - listener_range: IQR of YES'd artist listener counts → filter discovery results
      - booking_range: IQR of YES'd artist booking totals → filter out too-big / too-small
    """
    tag_counts: dict[str, int] = {}
    listener_vals: list[int] = []
    booking_vals: list[int] = []

    for name in yes_names:
        slug = _slug(name)
        enr = enriched_map.get(slug)
        if not enr:
            continue
        for tag in (enr.get("lastfm_tags") or []) + (enr.get("ra_genres") or []):
            tag_low = tag.lower()
            tag_counts[tag_low] = tag_counts.get(tag_low, 0) + 1
        gh = enr.get("growth_history") or {}
        lm = gh.get("current_listeners") or enr.get("spotify_followers")
        if lm:
            listener_vals.append(lm)
        t = (enr.get("booking_stats") or {}).get("total") or 0
        if t:
            booking_vals.append(t)

    top_tags = sorted(tag_counts, key=lambda k: tag_counts[k], reverse=True)[:5]

    if len(listener_vals) >= 3:
        import numpy as np
        p25 = int(np.percentile(listener_vals, 25))
        p75 = int(np.percentile(listener_vals, 75))
        listener_range = (max(500, p25 // 3), min(5_000_000, p75 * 4))
    else:
        listener_range = (500, 2_000_000)

    if len(booking_vals) >= 3:
        import numpy as np
        p25 = int(np.percentile(booking_vals, 25))
        p75 = int(np.percentile(booking_vals, 75))
        booking_range = (max(0, p25 // 3), min(5000, p75 * 4))
    else:
        booking_range = (0, 2000)

    return {"top_tags": top_tags, "listener_range": listener_range, "booking_range": booking_range}


def _collect_tag_artists(tags: list[str], limit_per_tag: int = 30) -> list[str]:
    """Fetch top artists from Last.fm for each genre tag derived from the Feel Matrix."""
    from lastfm_scraper import fetch_tag_top_artists
    seen: dict[str, None] = {}
    for tag in tags[:4]:   # cap at 4 tags — enough breadth without too many calls
        for name in fetch_tag_top_artists(tag, limit=limit_per_tag):
            seen.setdefault(name)
        time.sleep(0.25)
    return list(seen.keys())


def _fetch_lastfm_snapshot(name: str) -> dict | None:
    """Fetch LFM data for a new artist and append to LastFMSnapshot.jsonl."""
    try:
        from lastfm_scraper import fetch_artist_info, append_snapshot
        info = fetch_artist_info(name)
        if not info:
            return None
        snapshot = {
            "name":               info["name"],
            "query_name":         name,
            "listeners":          info["listeners"],
            "playcount":          info["playcount"],
            "plays_per_listener": round(info["playcount"] / info["listeners"], 2) if info["listeners"] else 0,
            "tags":               info["tags"],
            "similar":            info["similar"],
            "pf_fans":            0,
            "pf_past":            0,
            "pf_upcoming":        0,
            "pf_genres":          [],
            "scraped_at":         datetime.now(timezone.utc).isoformat(),
            "is_mainstream":      info["listeners"] > 1_500_000,
        }
        append_snapshot(snapshot)
        return snapshot
    except Exception:
        return None


def _minimal_enriched(name: str, snap: dict) -> dict:
    """Build a minimal enriched-style dict from a fresh LFM snapshot."""
    aid = _slug(snap.get("name") or name)
    return {
        "artist_id":    aid,
        "name":         snap.get("name") or name,
        "growth_history": {
            "current_listeners": snap.get("listeners"),
            "current_playcount": snap.get("playcount"),
            "snapshots": [{"date": snap["scraped_at"][:10],
                           "listeners": snap.get("listeners"),
                           "playcount": snap.get("playcount")}],
            "listener_delta_total": None,
            "listener_growth_pct_total": None,
            "days_tracked": 0,
        },
        "lastfm_tags":    snap.get("tags") or [],
        "lastfm_similar": snap.get("similar") or [],
        "booking_stats":  {},
        "momentum_score": 0,
    }


def _append_profile(profile) -> None:
    _PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROFILES_FILE, "a", encoding="utf-8") as f:
        f.write(profile.model_dump_json() + "\n")


def _append_candidate(artist_id: str, name: str, enriched: dict) -> None:
    _CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "artist_id": artist_id,
        "name":      name,
        "enriched":  enriched,
        "added_at":  datetime.now(timezone.utc).isoformat(),
        "source":    "discovery",
    }
    with open(_CANDIDATES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rank_candidates_by_lofi_feel(
    names: list[str],
    enriched_map: dict[str, dict],
    existing_slugs: set[str],
) -> list[str]:
    """
    Re-order names so those whose structured feature vector is closest to the
    LOFI feature centroid come first. Names not in enriched_map keep their
    original position (appended after the ranked known ones).
    """
    try:
        import numpy as np
        from lofi_tinder.embedder import extract_feature_vector, load_feature_centroid
        centroid = load_feature_centroid()
        if centroid is None:
            return names
        c_norm = np.linalg.norm(centroid)
        if c_norm == 0:
            return names

        known_scored: list[tuple[float, str]] = []
        unknown: list[str] = []

        for name in names:
            slug = _slug(name)
            if slug in existing_slugs:
                continue
            enr = enriched_map.get(slug)
            if enr:
                vec = extract_feature_vector(enr)
                v_norm = np.linalg.norm(vec)
                sim = float(np.dot(vec, centroid) / (v_norm * c_norm)) if v_norm > 0 else 0.0
                known_scored.append((sim, name))
            else:
                unknown.append(name)

        known_scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in known_scored] + unknown

    except Exception:
        return names


def discover_new_batch(
    yes_names: list[str],
    swiped_ids: set[str],
    profiles: dict,
    n: int = 20,
    progress_cb=None,
) -> list[str]:
    """
    Find n new artists via Last.fm similar-artist chains seeded from YES'd artists.

    Candidates are pre-sorted by feature similarity to the LOFI centroid so that
    the most LOFI-like artists (on structured features) are processed first.

    For each new artist:
      1. Fetch Last.fm data (adds a timestamped snapshot — builds time series)
      2. Build a profile via Claude API + sentence-transformer embedding
      3. Append to profiles.jsonl and candidates.jsonl

    Returns list of new artist_ids added to the pool.
    """
    from lofi_tinder.profile_builder import generate_profile_template
    from lofi_tinder.embedder import embed_profiles, load_centroid

    import numpy as np

    enriched_map   = _load_enriched_map()
    # Only block swiped artists — profiled artists can re-enter (their cached
    # profiles are reused instantly, no API call needed)
    existing_slugs = set(swiped_ids)

    # Seed rotation: prefer YES artists not recently used as Last.fm seeds.
    seed_history = _load_seed_history()
    seeds = _select_seeds(yes_names, seed_history, max_seeds=8)

    # Channel 1: relationship graph — artist.getSimilar (50 per seed)
    similar_names = _collect_similar_names(seeds)

    # Channel 2: tag graph — translate Feel Matrix top genre tags to candidate names
    # This is the core Feel Matrix → filter translation:
    # extract top tags from YES'd artists → search Last.fm by tag → new candidates
    filter_params = _extract_filter_params(yes_names, enriched_map)
    if filter_params["top_tags"]:
        tag_names = _collect_tag_artists(filter_params["top_tags"], limit_per_tag=30)
    else:
        tag_names = []

    # Combine both channels, deduplicate, filter already-swiped
    seen_combined: dict[str, None] = {}
    for nm in similar_names + tag_names:
        seen_combined.setdefault(nm)
    raw_new = [nm for nm in seen_combined if _slug(nm) not in existing_slugs]

    # Apply listener-range filter (derived from Feel Matrix) — skip artists clearly outside the range
    listener_min, listener_max = filter_params["listener_range"]
    def _in_range(name: str) -> bool:
        enr = enriched_map.get(_slug(name))
        if not enr:
            return True  # unknown — let through, will be fetched from Last.fm
        lm = (enr.get("growth_history") or {}).get("current_listeners") or enr.get("spotify_followers")
        if lm is None:
            return True
        return listener_min <= lm <= listener_max
    raw_new = [nm for nm in raw_new if _in_range(nm)]

    # Pre-sort by feature similarity to LOFI centroid (nearest-cluster):
    # Known artists are ranked by structured feature match; unknowns appended after.
    new_names = _rank_candidates_by_lofi_feel(raw_new, enriched_map, existing_slugs)[:n * 3]

    if not new_names:
        return []

    centroid = load_centroid()
    new_ids: list[str] = []
    total = min(len(new_names), n * 2)
    done  = 0

    for name in new_names:
        if len(new_ids) >= n:
            break

        slug = _slug(name)
        if slug in existing_slugs:
            continue

        if progress_cb:
            progress_cb(done, total, name)
        done += 1

        # Use stored enriched data if available, otherwise fetch from Last.fm
        enriched = enriched_map.get(slug)
        if not enriched:
            time.sleep(0.25)
            snap = _fetch_lastfm_snapshot(name)
            if not snap:
                continue
            enriched = _minimal_enriched(name, snap)
            canonical_name = snap.get("name") or name
        else:
            canonical_name = enriched["name"]
            slug = enriched["artist_id"]   # use canonical slug

        if slug in existing_slugs:
            continue

        # SHORT-CIRCUIT: already profiled — just re-queue, no Claude/embedding call
        if slug in profiles:
            _append_candidate(slug, canonical_name, enriched)
            existing_slugs.add(slug)
            new_ids.append(slug)
            continue

        from lofi_tinder.schemas import ArtistInput
        artist_input = ArtistInput(
            artist_id=slug,
            name=canonical_name,
            enriched=enriched,
        )

        try:
            profile = generate_profile_template(artist_input)
        except Exception:
            continue

        # Add embedding via sentence-transformers
        try:
            profiled = embed_profiles([profile])
            profile = profiled[0]
        except Exception:
            continue

        if not profile.embedding:
            continue

        # Compute cosine distance to current LOFI centroid
        if centroid is not None:
            vec  = np.array(profile.embedding, dtype="float64")
            norm = np.linalg.norm(vec)
            c_norm = np.linalg.norm(centroid)
            profile.cosine_dist_to_centroid = float(
                1.0 - (vec @ centroid) / (norm * c_norm)
            ) if norm > 0 and c_norm > 0 else 1.0

        _append_profile(profile)
        _append_candidate(slug, canonical_name, enriched)

        existing_slugs.add(slug)
        new_ids.append(slug)

    if progress_cb:
        progress_cb(len(new_ids), total, "done")

    # Record which artists were used as seeds so they're deprioritised next round
    if seeds and new_ids:
        _record_seeds_used(seeds)

    return new_ids
