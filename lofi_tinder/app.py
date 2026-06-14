"""
LOFI Tinder — Streamlit artist discovery UI.

Flow:
  select  →  scrape (live data, progress bars)  →  swipe (cards)
    ↑                                                    │
    └────────────── next batch ──────────────────────────┘

Usage:
    cd Testing/lofi-tinder
    streamlit run lofi_tinder/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Inject Streamlit Cloud secrets into os.environ BEFORE any credential reads
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass

from lofi_tinder.embedder import (
    compute_centroid, compute_feature_centroid, extract_feature_vector,
    load_centroid, save_centroid, save_feature_centroid,
)
from lofi_tinder.mab import LinUCB, reward_for_decision
from lofi_tinder.neo4j_client import get_client as _neo4j
from lofi_tinder.ranker import get_swiped_ids, load_swipes, rank_candidates
from lofi_tinder.schemas import ArtistProfile, SwipeRecord
from scrapers.unified_scraper import SOURCES, merge_into_enriched, scrape_batch

_NEGATIVE_DECISIONS = {"no", "commercial", "wrong_genre", "saturated_nl", "not_ready"}

_DATA_DIR     = Path(__file__).parent.parent / "data"
_PROFILES_FILE = Path(__file__).parent.parent / "profiles" / "artist_profiles.jsonl"
_SWIPES_FILE  = _DATA_DIR / "swipes.jsonl"
_CENTROID_UPDATE_EVERY = 20   # YES swipes before centroid refresh

st.set_page_config(page_title="LOFI Tinder", page_icon="🎛", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_profiles() -> dict[str, ArtistProfile]:
    profiles: dict[str, ArtistProfile] = {}
    if not _PROFILES_FILE.exists():
        return profiles
    for line in _PROFILES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            p = ArtistProfile(**data)
            if p.embedding:
                profiles[p.artist_id] = p
        except Exception:
            pass
    return profiles


@st.cache_data(ttl=300)
def _load_enriched_map() -> dict[str, dict]:
    enriched_file = Path(__file__).parent.parent / "scraper_data" / "artist_enriched.jsonl"
    result: dict[str, dict] = {}
    if not enriched_file.exists():
        return result
    for line in enriched_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            aid  = data.get("artist_id", "")
            result[aid] = data
            for old_id in data.get("old_artist_ids") or []:
                result[old_id] = data
        except Exception:
            pass
    return result


@st.cache_data(ttl=300)
def _load_candidates_map() -> dict[str, dict]:
    cfile  = _DATA_DIR / "candidates.jsonl"
    result = {}
    if not cfile.exists():
        return result
    for line in cfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                result[data.get("artist_id", "")] = data
            except Exception:
                pass
    return result


def _load_all_swipes() -> list[SwipeRecord]:
    """Load swipes from Neo4j (primary) falling back to local file."""
    neo4j = _neo4j()
    if neo4j.available:
        raw = neo4j.load_swipes()
        swipes = []
        for r in raw:
            try:
                swipes.append(SwipeRecord(
                    artist_id=r["artist_id"],
                    name=r.get("name", ""),
                    decision=r["decision"],
                    ts=r["ts"],
                    cosine_dist_at_swipe=r.get("score", 1.0),
                    linucb_score_at_swipe=0.0,
                    profile_text=r.get("profile_text", ""),
                ))
            except Exception:
                pass
        return swipes
    return load_swipes()   # fallback: local file


def _save_swipe(swipe: SwipeRecord) -> None:
    # Always write to local file (survives within session, backup for Cloud)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SWIPES_FILE, "a", encoding="utf-8") as f:
        f.write(swipe.model_dump_json() + "\n")
    # Also write to Neo4j if available
    neo4j = _neo4j()
    if neo4j.available:
        neo4j.save_swipe(
            artist_id=swipe.artist_id,
            name=swipe.name,
            decision=swipe.decision,
            ts=swipe.ts,
            score=swipe.cosine_dist_at_swipe,
            profile_text=swipe.profile_text,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Centroid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _update_centroid_from_swipes(swipes: list[SwipeRecord], profiles: dict[str, ArtistProfile]) -> None:
    yes_swipes = [
        s for s in swipes
        if s.decision == "yes" and s.artist_id in profiles and profiles[s.artist_id].embedding
    ]
    if not yes_swipes:
        return
    centroid = compute_centroid([profiles[s.artist_id].embedding for s in yes_swipes])
    save_centroid(centroid)
    emap = _load_enriched_map()
    feature_vecs = []
    for s in yes_swipes:
        enriched = _effective_enriched(s.artist_id, emap)
        if enriched:
            feature_vecs.append(extract_feature_vector(enriched))
    if feature_vecs:
        save_feature_centroid(compute_feature_centroid(feature_vecs))
    st.cache_data.clear()


def _effective_enriched(artist_id: str, emap: dict) -> dict:
    """Return enriched dict: prefer session fresh data, then file, then candidates."""
    fresh = st.session_state.get("batch_enriched_fresh") or {}
    if artist_id in fresh:
        return fresh[artist_id]
    if artist_id in emap:
        return emap[artist_id]
    cmap  = _load_candidates_map()
    cdata = cmap.get(artist_id, {})
    return cdata.get("enriched") or cdata


# ─────────────────────────────────────────────────────────────────────────────
# Swipe handling
# ─────────────────────────────────────────────────────────────────────────────

def _count_yes(swipes: list[SwipeRecord]) -> int:
    return sum(1 for s in swipes if s.decision == "yes")


def _count_neg(swipes: list[SwipeRecord]) -> int:
    return sum(1 for s in swipes if s.decision in _NEGATIVE_DECISIONS)


def _handle_swipe(
    artist: ArtistProfile,
    decision: str,
    mab: LinUCB,
    mab_scores: dict,
    emap: dict,
) -> None:
    record = SwipeRecord(
        artist_id=artist.artist_id,
        name=artist.name,
        decision=decision,
        ts=datetime.now(timezone.utc).isoformat(),
        cosine_dist_at_swipe=artist.cosine_dist_to_centroid,
        linucb_score_at_swipe=mab_scores.get(artist.artist_id, 0.0),
        profile_text=artist.profile_text,
    )
    _save_swipe(record)
    reward = reward_for_decision(decision)
    if reward is not None:
        enriched = _effective_enriched(artist.artist_id, emap)
        if enriched:
            import numpy as np
            mab.update(extract_feature_vector(enriched).astype("float64"), reward)
            mab.save()
    if decision != "skip":
        st.session_state["session_swiped"] = st.session_state.get("session_swiped", 0) + 1
    if decision == "yes":
        st.session_state["session_yes"] = st.session_state.get("session_yes", 0) + 1
        # Also store similar artists in Neo4j graph
        neo4j = _neo4j()
        if neo4j.available:
            enriched = _effective_enriched(artist.artist_id, emap)
            sims = list(dict.fromkeys(
                (enriched.get("lastfm_similar") or []) + (enriched.get("spotify_related") or [])
            ))
            if sims:
                neo4j.save_similar_edges(artist.artist_id, sims)
    elif decision == "monitor":
        st.session_state["session_monitor"] = st.session_state.get("session_monitor", 0) + 1
    elif decision in _NEGATIVE_DECISIONS:
        st.session_state["session_no"] = st.session_state.get("session_no", 0) + 1
    st.session_state["queue_idx"] = st.session_state.get("queue_idx", 0) + 1
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Phase: SELECT  — pick next 20 candidates
# ─────────────────────────────────────────────────────────────────────────────

def _phase_select(
    profiles: dict[str, ArtistProfile],
    swipes: list[SwipeRecord],
    emap: dict,
    mab: LinUCB,
    mab_scores: dict,
) -> None:
    swiped_ids = get_swiped_ids(swipes)
    queue = rank_candidates(profiles, swiped_ids, emap, mab_scores, limit=20)
    if not queue:
        st.warning("No unseen candidates in the pool. Run `python run.py --candidates` to add more.")
        return
    st.session_state["current_batch"]    = queue
    st.session_state["batch_scraped"]    = False
    st.session_state["batch_enriched_fresh"] = {}
    st.session_state["queue_idx"]        = 0
    st.session_state["phase"]            = "scrape"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Phase: SCRAPE  — live-scrape the 20 artists with per-source progress bars
# ─────────────────────────────────────────────────────────────────────────────

def _phase_scrape(emap: dict) -> None:
    batch: list[ArtistProfile] = st.session_state.get("current_batch", [])
    if not batch:
        st.session_state["phase"] = "select"
        st.rerun()
        return

    names = [a.name for a in batch]
    n     = len(names)

    st.title("🔍 Gathering fresh data…")
    st.caption(f"Checking {n} artists across {len(SOURCES)} sources before swiping")

    # Create one progress bar per source
    bars: dict[str, st.delta_generator.DeltaGenerator] = {}
    for src in SOURCES:
        bars[src] = st.progress(0.0, text=f"⏳  {src}")

    # Status line below the bars
    status = st.empty()

    def _cb(source: str, done: int, total: int, name: str) -> None:
        pct = done / total
        bars[source].progress(pct, text=f"{'✓' if done==total else '⟳'}  {source}  —  {name}")
        status.caption(f"{source}: {done}/{total}")

    # Run the scraping (synchronous — progress bars update live via Streamlit delta)
    raw_results = scrape_batch(names, progress_cb=_cb)

    # Mark all bars complete
    for src in SOURCES:
        bars[src].progress(1.0, text=f"✓  {src}  complete")
    status.empty()

    # Merge fresh data into enriched records and store in session
    fresh: dict[str, dict] = {}
    for artist in batch:
        base    = emap.get(artist.artist_id) or {}
        merged  = merge_into_enriched(base, raw_results.get(artist.name, {}))
        merged["artist_id"] = artist.artist_id
        merged["name"]      = artist.name
        fresh[artist.artist_id] = merged

        # Persist to Neo4j
        neo4j = _neo4j()
        if neo4j.available:
            neo4j.upsert_artist(artist.artist_id, {
                "name":             artist.name,
                "spotify_followers":merged.get("spotify_followers"),
                "yt_subscribers":   merged.get("yt_subscribers"),
                "mc_followers":     merged.get("mc_followers"),
                "discogs_releases": merged.get("discogs_releases"),
                "profile_text":     artist.profile_text,
            })

    st.session_state["batch_enriched_fresh"] = fresh
    st.session_state["batch_scraped"]        = True
    st.session_state["phase"]               = "swipe"
    st.session_state["queue_idx"]           = 0
    time.sleep(0.3)   # brief pause so user sees "✓ complete" state
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Phase: SWIPE  — the main card-swiping UI
# ─────────────────────────────────────────────────────────────────────────────

def _phase_swipe(
    profiles: dict[str, ArtistProfile],
    swipes: list[SwipeRecord],
    emap: dict,
    mab: LinUCB,
    mab_scores: dict,
) -> None:
    queue: list[ArtistProfile] = st.session_state.get("current_batch", [])
    if not queue:
        st.session_state["phase"] = "select"
        st.rerun()
        return

    idx: int = st.session_state.get("queue_idx", 0)

    # ── Stats bar ────────────────────────────────────────────────────────────
    _ss_swiped  = st.session_state.get("session_swiped",  0)
    _ss_yes     = st.session_state.get("session_yes",     0)
    _ss_monitor = st.session_state.get("session_monitor", 0)
    _ss_no      = st.session_state.get("session_no",      0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reviewed (session)", _ss_swiped)
    c2.metric("YES",     _ss_yes)
    c3.metric("Monitor", _ss_monitor)
    c4.metric("No",      _ss_no)
    c5.metric("Remaining", max(0, len(queue) - idx))

    # Centroid update progress
    next_update = _CENTROID_UPDATE_EVERY - (_ss_yes % _CENTROID_UPDATE_EVERY)
    if _ss_yes > 0 and next_update == _CENTROID_UPDATE_EVERY:
        st.success(f"Taste profile updated! ({_ss_yes} YES swipes this session)")
    else:
        st.info(f"Taste profile update in {next_update} more YES swipe(s)")

    if _ss_yes > 0 and _ss_yes % _CENTROID_UPDATE_EVERY == 0:
        if st.session_state.get("last_centroid_update") != _ss_yes:
            _update_centroid_from_swipes(swipes, profiles)
            mab.save()
            st.session_state["last_centroid_update"] = _ss_yes
            st.session_state["queue_stale"] = True

    # ── Batch complete ────────────────────────────────────────────────────────
    if idx >= len(queue):
        _show_batch_end(swipes, profiles, get_swiped_ids(swipes), mab, mab_scores)
        return

    # ── Current card ─────────────────────────────────────────────────────────
    artist = queue[idx]
    dist   = artist.cosine_dist_to_centroid

    # Merge session fresh data into display enriched (prefer fresh over stale file)
    _fresh = st.session_state.get("batch_enriched_fresh") or {}
    display_enriched = _fresh.get(artist.artist_id) or emap.get(artist.artist_id) or {}

    st.divider()

    # Header: photo + name
    img_url = display_enriched.get("image_url") or _fetch_spotify_image(artist.name)
    if img_url:
        img_html = (
            f'<img src="{img_url}" '
            f'style="width:96px;height:96px;border-radius:50%;object-fit:cover;'
            f'border:2px solid #444;flex-shrink:0">'
        )
    else:
        initials = "".join(p[0].upper() for p in artist.name.split()[:2] if p)[:2]
        palette  = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]
        bg       = palette[sum(ord(c) for c in artist.name) % len(palette)]
        img_html = (
            f'<div style="width:96px;height:96px;border-radius:50%;background:{bg};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:32px;font-weight:700;color:white;flex-shrink:0">{initials}</div>'
        )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:20px;padding:8px 0 4px 0">'
        f'{img_html}'
        f'<div style="flex:1;min-width:0">'
        f'<h2 style="margin:0;font-size:2em;font-weight:700;line-height:1.1">{artist.name}</h2>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    _show_stats(
        artist.artist_id, artist.profile_text, dist,
        nearest_cluster=getattr(artist, "nearest_cluster", "unknown"),
        enriched_override=display_enriched,
    )

    st.divider()

    # Swipe buttons
    b_yes, b_monitor, b_skip = st.columns([3, 3, 1])
    with b_yes:
        if st.button("YES — Fits LOFI", use_container_width=True, type="primary", key="swipe_yes"):
            _handle_swipe(artist, "yes", mab, mab_scores, emap)
    with b_monitor:
        if st.button("MONITOR — Interesting, not yet", use_container_width=True, key="swipe_monitor"):
            _handle_swipe(artist, "monitor", mab, mab_scores, emap)
    with b_skip:
        if st.button("Skip", use_container_width=True, key="swipe_skip", help="No signal recorded"):
            _handle_swipe(artist, "skip", mab, mab_scores, emap)

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        if st.button("No fit",        use_container_width=True, type="secondary", key="swipe_no"):
            _handle_swipe(artist, "no",            mab, mab_scores, emap)
    with r2:
        if st.button("Too commercial", use_container_width=True, type="secondary", key="swipe_commercial"):
            _handle_swipe(artist, "commercial",    mab, mab_scores, emap)
    with r3:
        if st.button("Wrong genre",   use_container_width=True, type="secondary", key="swipe_genre"):
            _handle_swipe(artist, "wrong_genre",   mab, mab_scores, emap)
    with r4:
        if st.button("Saturated NL",  use_container_width=True, type="secondary", key="swipe_saturated"):
            _handle_swipe(artist, "saturated_nl",  mab, mab_scores, emap)
    with r5:
        if st.button("Not ready yet", use_container_width=True, type="secondary", key="swipe_notready"):
            _handle_swipe(artist, "not_ready",     mab, mab_scores, emap)


# ─────────────────────────────────────────────────────────────────────────────
# Batch end — shown when all 20 are swiped
# ─────────────────────────────────────────────────────────────────────────────

def _show_batch_end(
    swipes: list[SwipeRecord],
    profiles: dict,
    swiped_ids: set,
    mab: LinUCB,
    mab_scores: dict,
) -> None:
    last_batch    = swipes[-20:] if len(swipes) >= 20 else swipes
    yes_names     = [s.name for s in last_batch if s.decision == "yes"]
    monitor_names = [s.name for s in last_batch if s.decision == "monitor"]
    no_names      = [s.name for s in last_batch if s.decision in _NEGATIVE_DECISIONS]
    skip_names    = [s.name for s in last_batch if s.decision == "skip"]

    st.subheader("Batch complete")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("YES",     len(yes_names))
    c2.metric("Monitor", len(monitor_names))
    c3.metric("No",      len(no_names))
    c4.metric("Skip",    len(skip_names))
    if yes_names:
        st.markdown("**YES'd:** " + " · ".join(yes_names[:10]))
    if monitor_names:
        st.caption("Monitoring: " + " · ".join(monitor_names[:10]))
    if no_names:
        reason_counts: dict[str, int] = {}
        for s in last_batch:
            if s.decision in _NEGATIVE_DECISIONS:
                reason_counts[s.decision] = reason_counts.get(s.decision, 0) + 1
        if reason_counts:
            st.caption("Rejections: " + "  ·  ".join(
                f"{k.replace('_',' ')} ({v})" for k, v in reason_counts.items()
            ))

    neo4j = _neo4j()
    if neo4j.available:
        counts = neo4j.count_swipes()
        total_yes = counts.get("yes", 0)
        st.caption(f"Neo4j: {sum(counts.values())} total swipes saved  ·  {total_yes} YES overall")

    st.divider()
    st.markdown(
        "LOFI Feel Matrix updated from your swipes. "
        "The next 20 artists are selected and scraped based on your taste profile."
    )

    if st.button("Find next 20 artists", type="primary", use_container_width=True):
        st.session_state["phase"] = "select"
        st.session_state.pop("current_batch", None)
        st.session_state["queue_stale"] = True
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎛 LOFI Artist Tinder")

    profiles = _load_profiles()
    if not profiles:
        st.error("No artist profiles found. Run: `python run.py --seed && python run.py --candidates`")
        return

    swipes     = _load_all_swipes()
    swiped_ids = get_swiped_ids(swipes)
    yes_count  = _count_yes(swipes)

    emap = _load_enriched_map()

    mab = LinUCB.load()
    if len(swipes) >= 20:
        feature_vecs: dict[str, list[float]] = {}
        for aid, profile in profiles.items():
            enr = emap.get(aid) or {}
            if enr:
                feature_vecs[aid] = extract_feature_vector(enr).tolist()
        mab_scores = mab.score_batch(feature_vecs)
    else:
        mab_scores = {}

    # Phase state machine
    phase = st.session_state.get("phase", "select")

    neo4j = _neo4j()
    if neo4j.available:
        st.sidebar.success("Neo4j connected")
    else:
        st.sidebar.warning("Neo4j not connected — swipes saved locally only")

    if phase == "select":
        _phase_select(profiles, swipes, emap, mab, mab_scores)
    elif phase == "scrape":
        _phase_scrape(emap)
    elif phase == "swipe":
        _phase_swipe(profiles, swipes, emap, mab, mab_scores)
    else:
        st.session_state["phase"] = "select"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Spotify image, KV grid, growth chart
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _fetch_spotify_image(artist_name: str) -> str | None:
    import base64, re, urllib.request, urllib.parse
    cid = os.environ.get("SPOTIFY_CLIENT_ID", "")
    sec = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not cid or not sec:
        return None
    clean = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", artist_name).strip()
    try:
        creds = base64.b64encode(f"{cid}:{sec}".encode()).decode()
        req = urllib.request.Request(
            "https://accounts.spotify.com/api/token",
            data=b"grant_type=client_credentials",
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            token = json.loads(r.read()).get("access_token")
        if not token:
            return None
        def _search(name):
            q = urllib.parse.quote(name)
            req2 = urllib.request.Request(
                f"https://api.spotify.com/v1/search?q={q}&type=artist&limit=3",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req2, timeout=5) as r:
                data = json.loads(r.read())
            items = (data.get("artists") or {}).get("items") or []
            if items:
                imgs = items[0].get("images") or []
                return imgs[min(1, len(imgs)-1)].get("url") if imgs else None
            return None
        result = _search(clean)
        if not result and clean != artist_name:
            result = _search(artist_name)
        return result
    except Exception:
        return None


def _kv_grid(items: list[tuple[str, str]]) -> None:
    items = [(l, v) for l, v in items if v not in (None, "", "0", 0)]
    if not items:
        return
    cells = "".join(
        f'<div style="min-width:80px">'
        f'<div style="font-size:0.68em;color:#9ca3af;text-transform:uppercase;'
        f'letter-spacing:0.4px;margin-bottom:2px">{label}</div>'
        f'<div style="font-size:0.95em;font-weight:600;color:#f3f4f6">{value}</div>'
        f'</div>'
        for label, value in items
    )
    st.markdown(
        f'<div style="display:flex;gap:28px;flex-wrap:wrap;padding:6px 0 4px 0">{cells}</div>',
        unsafe_allow_html=True,
    )


def _plot_growth(snapshots: list[dict], y_col: str = "Listeners", color: str = "#4ade80") -> None:
    if len(snapshots) < 2:
        return
    try:
        import pandas as pd, altair as alt
        rows = [{"date": s["date"], y_col: s.get("listeners") or s.get(y_col)}
                for s in snapshots if s.get("date") and (s.get("listeners") or s.get(y_col))]
        rows = [r for r in rows if r[y_col] is not None]
        if len(rows) < 2:
            return
        df = (pd.DataFrame(rows).drop_duplicates("date")
              .assign(date=lambda x: pd.to_datetime(x["date"])).sort_values("date"))
        chart = (
            alt.Chart(df).mark_area(
                line={"color": color, "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear", x1=0, x2=0, y1=1, y2=0,
                    stops=[alt.GradientStop(color=color, offset=1),
                           alt.GradientStop(color="transparent", offset=0)],
                ),
            ).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)),
                y=alt.Y(f"{y_col}:Q", title=None, axis=alt.Axis(format=",.0f")),
            ).properties(height=150)
        )
        st.altair_chart(chart, width='stretch')
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Score computation (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

_MILESTONE_LABELS = {
    "first_ibiza": "First Ibiza booking", "first_circoloco": "First Circoloco",
    "first_music_on": "First Music On",   "first_ants": "First ANTS",
    "first_piv_release": "First PIV release",
    "first_beatport_top10": "First Beatport Top 10",
    "first_beatport_no1": "First Beatport #1",
    "first_festival": "First festival",
    "first_boiler_room": "First Boiler Room",
    "first_ra_podcast": "First RA Podcast",
    "first_bbc_r1": "First BBC Radio 1",
    "first_headline_500": "First headline 500+",
    "first_headline_1000": "First headline 1,000+",
    "first_headline_2000": "First headline 2,000+",
    "first_headline_5000": "First headline 5,000+",
    "first_tier_a_support": "First Tier A support",
    "first_tier_a_b2b": "First Tier A B2B",
    "first_extended_set": "First extended set",
    "first_anl": "First All Night Long",
    "first_adl": "First All Day Long",
    "first_major_residency": "First major residency",
    "first_multi_city_tour": "First multi-city tour",
}

_HIGH_SIGNAL_MILESTONES = {
    "first_circoloco", "first_music_on", "first_ants",
    "first_beatport_top10", "first_beatport_no1",
    "first_boiler_room", "first_ra_podcast", "first_bbc_r1",
    "first_headline_1000", "first_headline_2000", "first_headline_5000",
    "first_tier_a_b2b",
}

_NOTABLE_LABELS = {
    "Solid Grooves Records", "PIV Records", "Up The Stuss", "Hot Creations",
    "Cuttin Headz", "Revival New York", "No Art", "Eastenderz",
    "Heavy House Society", "Cecille Records", "CircoLoco Records", "Afterlife",
    "Diynamic Music", "Innervisions", "Kompakt", "Drumcode", "Tronic",
    "Soma Records", "Repitch Recordings", "Stroboscopic Artefacts",
}


def _compute_card_scores(enriched: dict, cosine_dist: float) -> dict:
    import math
    bs  = enriched.get("booking_stats") or {}
    gh  = enriched.get("growth_history") or {}
    total     = bs.get("total") or 0
    recent_12 = bs.get("recent_12m") or 0
    vel       = bs.get("booking_velocity") or 0.0
    nl_ratio  = bs.get("nl_ratio") or enriched.get("nl_ratio") or 0.0
    geo       = bs.get("geo_spread") or enriched.get("geo_spread") or 0
    nl_events = int(round((bs.get("nl_events") or 0) or recent_12 * nl_ratio))
    bp_tier   = enriched.get("beatport_label_tier")
    festivals = enriched.get("festival_history") or []
    mc        = enriched.get("mixcloud_appearances") or 0
    milestones= enriched.get("milestones") or {}
    listeners  = gh.get("current_listeners") or enriched.get("spotify_followers") or 0
    pf_fans    = enriched.get("pf_fans") or 0
    ra_ev      = enriched.get("ra_genre_events") or 0

    sound_fit = max(0, min(100, int((1 - cosine_dist) * 100)))

    if recent_12 == 0:     book_pts = 0
    elif recent_12 <= 2:   book_pts = 8
    elif recent_12 <= 5:   book_pts = 16
    elif recent_12 <= 10:  book_pts = 22
    elif recent_12 <= 20:  book_pts = 28
    elif recent_12 <= 50:  book_pts = 34
    else:                  book_pts = 40

    if not vel:            vel_pts = 8
    elif vel >= 2.0:       vel_pts = 20
    elif vel >= 1.5:       vel_pts = 17
    elif vel >= 1.2:       vel_pts = 14
    elif vel >= 1.0:       vel_pts = 11
    elif vel >= 0.7:       vel_pts = 7
    elif vel >= 0.4:       vel_pts = 3
    else:                  vel_pts = 0

    if geo == 0:           geo_pts = 0
    elif geo == 1:         geo_pts = 4
    elif geo <= 3:         geo_pts = 8
    elif geo <= 7:         geo_pts = 12
    elif geo <= 14:        geo_pts = 16
    else:                  geo_pts = 20

    aud = max(listeners, pf_fans * 20)
    if aud == 0:           aud_pts = 0
    elif aud < 1_000:      aud_pts = 3
    elif aud < 10_000:     aud_pts = 7
    elif aud < 50_000:     aud_pts = 11
    elif aud < 200_000:    aud_pts = 15
    elif aud < 1_000_000:  aud_pts = 18
    else:                  aud_pts = 20

    heat   = min(100, book_pts + vel_pts + geo_pts + aud_pts)
    nl_sat_pts = max(0, 60 - nl_events * 10)
    vel_rising = min(40, int(max(0, vel - 1.0) * 40))
    window = min(100, nl_sat_pts + vel_rising)

    nf = len(festivals)
    if nf == 0:    fest_pts = 0
    elif nf == 1:  fest_pts = 5
    elif nf <= 3:  fest_pts = 10
    elif nf <= 7:  fest_pts = 17
    elif nf <= 14: fest_pts = 23
    elif nf <= 20: fest_pts = 27
    else:          fest_pts = 30

    bp_pts = {"A+": 25, "A": 18, "B": 10}.get(bp_tier or "", 0)

    if total == 0:    depth_pts = 0
    elif total < 10:  depth_pts = 4
    elif total < 30:  depth_pts = 8
    elif total < 75:  depth_pts = 12
    elif total < 150: depth_pts = 16
    elif total < 300: depth_pts = 20
    elif total < 500: depth_pts = 23
    else:             depth_pts = 25

    ra_pts  = min(8, int(math.log10(max(ra_ev, 1)) / math.log10(201) * 8))
    mc_pts  = min(6, mc)
    ms_pts  = min(10, sum(5 for k in _HIGH_SIGNAL_MILESTONES if milestones.get(k)))
    ind_pts = min(20, ra_pts + mc_pts + ms_pts)
    track_record = min(100, fest_pts + bp_pts + depth_pts + ind_pts)

    if total >= 400 or (total >= 200 and bp_tier in ("A+", "A")):
        stage, stage_bg = "Established", "#6366f1"
    elif total >= 80 or (total >= 40 and vel >= 1.3):
        stage, stage_bg = "Rising",      "#16a34a"
    elif total >= 15:
        stage, stage_bg = "Emerging",    "#d97706"
    else:
        stage, stage_bg = "Underground", "#475569"

    if nl_events >= 8:
        nl_label, nl_bg = "Saturated NL",    "#dc2626"
    elif nl_events >= 4:
        nl_label, nl_bg = "Active in NL",    "#d97706"
    elif nl_events >= 1:
        nl_label, nl_bg = "Low NL presence", "#16a34a"
    else:
        nl_label, nl_bg = "Fresh to NL",     "#16a34a"

    return {
        "sound_fit": sound_fit, "heat": heat, "window": window,
        "track_record": track_record,
        "stage": stage, "stage_bg": stage_bg,
        "nl_label": nl_label, "nl_bg": nl_bg, "nl_events": nl_events,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Card display
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _load_label_artists_map() -> dict[str, list[str]]:
    v2   = Path(__file__).parent.parent.parent / "v2-scraper" / "scraper"
    path = v2 / "BeatportLabelArtistItem.jsonl"
    by_label: dict[str, list[str]] = {}
    if not path.exists():
        return by_label
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row   = json.loads(line)
            label = (row.get("label_name") or "").lower()
            artist = row.get("artist_name") or ""
            if label and artist:
                by_label.setdefault(label, [])
                if artist not in by_label[label]:
                    by_label[label].append(artist)
        except Exception:
            pass
    return by_label


def _show_stats(
    artist_id: str,
    profile_text: str,
    cosine_dist: float = 1.0,
    nearest_cluster: str = "unknown",
    enriched_override: dict | None = None,
) -> None:
    if enriched_override is not None:
        enriched = enriched_override
    else:
        emap = _load_enriched_map()
        enriched = emap.get(artist_id)
        if enriched is None:
            cmap  = _load_candidates_map()
            cdata = cmap.get(artist_id, {})
            enriched = cdata.get("enriched") or cdata
    if not enriched:
        if profile_text:
            st.info(profile_text)
        return

    gh    = enriched.get("growth_history") or {}
    bs    = enriched.get("booking_stats") or {}
    snaps = gh.get("snapshots") or []
    scores = _compute_card_scores(enriched, cosine_dist)

    # Pills
    def _pill(value, label, bg, wide=False):
        w = "min-width:90px" if wide else "min-width:72px"
        return (
            f'<div style="{w};padding:10px 14px;border-radius:10px;background:{bg};'
            f'color:white;text-align:center;display:inline-block">'
            f'<div style="font-size:1.5em;font-weight:800;line-height:1">{value}</div>'
            f'<div style="font-size:0.68em;margin-top:3px;opacity:0.9;letter-spacing:.5px">'
            f'{label.upper()}</div></div>'
        )
    cluster_bg  = {"core": "#4f46e5", "emerging": "#0d9488"}.get(nearest_cluster, "#475569")
    cluster_lbl = {"core": "Core",    "emerging": "Emerging"}.get(nearest_cluster, "?")
    pills_html  = " ".join([
        _pill(scores["stage"],    "Career Stage", scores["stage_bg"], wide=True),
        _pill(scores["nl_label"], "NL Status",    scores["nl_bg"],    wide=True),
        _pill(cluster_lbl,        "Cluster",      cluster_bg,         wide=True),
    ])
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 4px 0">{pills_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How are these labels determined?", expanded=False):
        st.markdown(
            "**Career Stage** — total bookings and Beatport label tier.\n\n"
            "**NL Status** — estimated NL bookings/yr from Partyflock data.\n\n"
            "**Cluster** — feature similarity to LOFI's *Core* (established) or *Emerging* booker profiles."
        )

    # Data coverage
    _cov: list[str] = []
    if gh.get("current_listeners"):
        _cov.append("Last.fm")
    if enriched.get("beatport_releases") or enriched.get("beatport_labels"):
        _cov.append("Beatport")
    if (bs.get("festival_count") or enriched.get("festival_history")):
        nf = bs.get("festival_count") or len(enriched.get("festival_history") or [])
        _cov.append(f"Festival lineups ({nf})")
    if bs.get("total"):
        nl = bs.get("nl_events") or 0
        _cov.append(f"Club lineups ({bs['total']}, {nl} NL)")
    if enriched.get("ra_genre_events") or enriched.get("ra_genres"):
        _cov.append("Resident Advisor")
    if enriched.get("spotify_id"):
        sf = enriched.get("spotify_followers")
        _cov.append(f"Spotify ({sf:,} followers)" if sf else "Spotify")
    if enriched.get("discogs_id"):
        dr = enriched.get("discogs_releases")
        _cov.append(f"Discogs ({dr} releases)" if dr else "Discogs")
    if enriched.get("yt_channel_id"):
        ys = enriched.get("yt_subscribers")
        _cov.append(f"YouTube ({ys:,} subs)" if ys else "YouTube")
    if enriched.get("mc_followers") or enriched.get("mixcloud_appearances"):
        _cov.append("Mixcloud")
    if enriched.get("sc_followers") or enriched.get("sc_tracks"):
        _cov.append("SoundCloud")
    if _cov:
        st.caption("Scraped: " + "  ·  ".join(_cov))

    bp_labels = enriched.get("beatport_labels") or []
    bp_lower  = {lb.lower() for lb in bp_labels}
    matched   = [lb for lb in _NOTABLE_LABELS if lb.lower() in bp_lower]
    if matched:
        st.caption("Notable labels: " + "  ·  ".join(matched))

    # Genre + similar
    tags = list(dict.fromkeys(
        (enriched.get("lastfm_tags") or []) +
        (enriched.get("ra_genres") or []) +
        (enriched.get("spotify_genres") or [])
    ))
    similar = list(dict.fromkeys(
        (enriched.get("lastfm_similar") or []) +
        (enriched.get("spotify_related") or [])
    ))
    if tags:
        st.caption("  ·  ".join(f"#{t}" for t in tags[:8]))
    if similar:
        st.caption(f"Similar to: {', '.join(similar[:8])}")

    # Key signal line
    bp_tier   = enriched.get("beatport_label_tier")
    agency    = enriched.get("agency")
    milestones= enriched.get("milestones") or {}
    signals: list[str] = []
    if agency:
        tier_s = f" `{enriched.get('agency_tier')}`" if enriched.get("agency_tier") else ""
        signals.append(f"**Agency:** {agency}{tier_s}")
    if bp_labels and bp_tier:
        signals.append(f"**Label:** {bp_labels[0]} `{bp_tier}`")
    achieved_hs = {k: v for k, v in milestones.items() if v and k in _HIGH_SIGNAL_MILESTONES}
    if achieved_hs:
        k, v = next(iter(achieved_hs.items()))
        signals.append(f"**{_MILESTONE_LABELS.get(k, k)}** `{v}`")
    if signals:
        st.markdown("  |  ".join(signals))

    if profile_text:
        st.info(profile_text)

    st.divider()

    # ── Data sections ─────────────────────────────────────────────────────────

    total    = bs.get("total") or 0
    recent12 = bs.get("recent_12m") or 0
    vel      = bs.get("booking_velocity")
    geo      = bs.get("geo_spread") or 0
    nl_ratio = bs.get("nl_ratio")
    pf_fans  = enriched.get("pf_fans") or 0
    all_evs  = bs.get("recent_events") or []
    fh       = enriched.get("festival_history") or []
    countries= bs.get("countries") or []

    if total or pf_fans or all_evs:
        with st.container(border=True):
            st.markdown("**Partyflock**")
            vel_str = None
            if vel:
                arrow = "↗" if vel > 1.1 else "↘" if vel < 0.9 else "→"
                vel_str = f"{vel:.1f}× {arrow}"
            _kv_grid([
                ("Fans",     f"{pf_fans:,}" if pf_fans else None),
                ("Career",   str(total) if total else None),
                ("Last 12m", str(recent12) if recent12 else None),
                ("Velocity", vel_str),
                ("NL ratio", f"{nl_ratio:.0%}" if nl_ratio else None),
                ("Countries","  ·  ".join(countries[:12]) if countries else None),
            ])
            if fh:
                st.caption("Festivals: " + "  ·  ".join(fh[:20]))
            today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            upcoming = sorted([e for e in all_evs if (e.get("date") or "") >= today], key=lambda x: x["date"])
            past     = sorted([e for e in all_evs if (e.get("date") or "") <  today], key=lambda x: x["date"], reverse=True)
            if upcoming or past:
                ev1, ev2 = st.columns(2)
                with ev1:
                    if upcoming:
                        st.markdown("**Upcoming**")
                        for ev in upcoming[:5]:
                            nm = ev.get("event_name") or ev.get("venue") or ""
                            ct = ev.get("city") or ""
                            st.markdown(f"`{ev['date']}` — {nm}{', ' + ct if ct else ''}")
                with ev2:
                    if past:
                        st.markdown("**Recent**")
                        for ev in past[:5]:
                            nm = ev.get("event_name") or ev.get("venue") or ""
                            ct = ev.get("city") or ""
                            st.markdown(f"`{ev['date']}` — {nm}{', ' + ct if ct else ''}")

    lfm_listeners = gh.get("current_listeners")
    lfm_playcount = gh.get("current_playcount")
    lfm_growth    = gh.get("listener_growth_pct_total")
    lfm_tags      = enriched.get("lastfm_tags") or []
    lfm_similar   = enriched.get("lastfm_similar") or []
    if lfm_listeners or lfm_tags:
        with st.container(border=True):
            st.markdown("**Last.fm**")
            ppl = round(lfm_playcount / lfm_listeners, 1) if lfm_listeners and lfm_playcount else None
            _kv_grid([
                ("Listeners",      f"{lfm_listeners:,}" if lfm_listeners else None),
                ("Growth",         f"{lfm_growth:+.1f}%" if lfm_growth is not None else None),
                ("Playcount",      f"{lfm_playcount:,}" if lfm_playcount else None),
                ("Plays/listener", f"{ppl}" if ppl else None),
                ("Tags",           "  ·  ".join(f"#{t}" for t in lfm_tags[:6]) if lfm_tags else None),
                ("Similar",        ", ".join(lfm_similar[:6]) if lfm_similar else None),
            ])
            unique_dates = {(s.get("date") or "")[:10] for s in snaps if s.get("listeners")}
            if len(unique_dates) >= 2:
                _plot_growth(snaps)
            elif snaps:
                st.caption("1 snapshot — run lastfm_enricher weekly to build growth history")

    sp_followers = enriched.get("spotify_followers")
    sp_pop       = enriched.get("spotify_popularity")
    sp_genres    = enriched.get("spotify_genres") or []
    sp_related   = enriched.get("spotify_related") or []
    sp_url       = enriched.get("spotify_url")
    if sp_followers or sp_genres or sp_related:
        with st.container(border=True):
            st.markdown("**Spotify**")
            _kv_grid([
                ("Followers",  f"{sp_followers:,}" if sp_followers else None),
                ("Popularity", f"{sp_pop}/100" if sp_pop else None),
                ("Genres",     "  ·  ".join(sp_genres[:6]) if sp_genres else None),
                ("Related",    ", ".join(sp_related[:6]) if sp_related else None),
            ])
            if sp_url:
                st.markdown(f"[Open on Spotify ↗]({sp_url})")

    bp_releases = enriched.get("beatport_releases")
    bp_latest   = enriched.get("beatport_latest_release")
    if bp_releases or bp_labels:
        with st.container(border=True):
            st.markdown("**Beatport**")
            _kv_grid([
                ("Releases",   str(bp_releases) if bp_releases else None),
                ("Label tier", bp_tier if bp_tier else None),
                ("Latest",     bp_latest if bp_latest else None),
                ("Labels",     ", ".join(bp_labels[:5]) if bp_labels else None),
            ])
            lmap = _load_label_artists_map()
            coartists: list[str] = []
            seen = {(enriched.get("name") or "").lower()}
            for lbl in bp_labels[:3]:
                for a in lmap.get(lbl.lower(), [])[:15]:
                    if a.lower() not in seen:
                        coartists.append(a); seen.add(a.lower())
            if coartists[:8]:
                st.caption("Label mates: " + "  ·  ".join(coartists[:8]))

    sc_followers = enriched.get("sc_followers")
    sc_tracks    = enriched.get("sc_tracks")
    sc_url       = enriched.get("sc_url")
    sc_snaps     = enriched.get("sc_snapshots") or []
    if sc_followers or sc_tracks:
        with st.container(border=True):
            st.markdown("**SoundCloud**")
            sc_growth = None
            if len(sc_snaps) >= 2:
                f0 = sc_snaps[0].get("followers") or 0
                fl = sc_snaps[-1].get("followers") or 0
                if f0:
                    sc_growth = f"{(fl - f0) / f0 * 100:+.1f}%"
            _kv_grid([
                ("Followers", f"{sc_followers:,}" if sc_followers else None),
                ("Growth",    sc_growth),
                ("Tracks",    str(sc_tracks) if sc_tracks else None),
            ])
            sc_unique = {s.get("date","") for s in sc_snaps if s.get("followers")}
            if len(sc_unique) >= 2:
                _plot_growth(
                    [{"date": s["date"], "listeners": s["followers"]} for s in sc_snaps
                     if s.get("date") and s.get("followers")],
                    y_col="Followers", color="#f97316",
                )
            if sc_url:
                st.markdown(f"[Open on SoundCloud ↗]({sc_url})")

    ra_ev     = enriched.get("ra_genre_events") or 0
    ra_genres = enriched.get("ra_genres") or []
    ra_cities = enriched.get("ra_cities") or []
    if ra_ev or ra_genres:
        with st.container(border=True):
            st.markdown("**Resident Advisor**")
            _kv_grid([
                ("Genre events", str(ra_ev) if ra_ev else None),
                ("Genres",       ", ".join(ra_genres[:6]) if ra_genres else None),
                ("Cities",       ", ".join(ra_cities[:6]) if ra_cities else None),
            ])

    dg_releases   = enriched.get("discogs_releases")
    dg_labels     = enriched.get("discogs_labels") or []
    dg_styles     = enriched.get("discogs_styles") or []
    dg_first_year = enriched.get("discogs_first_year")
    dg_url        = enriched.get("discogs_url") or (
        f"https://www.discogs.com/artist/{enriched['discogs_id']}"
        if enriched.get("discogs_id") else None
    )
    if dg_releases or dg_labels:
        with st.container(border=True):
            st.markdown("**Discogs**")
            _kv_grid([
                ("Releases", str(dg_releases) if dg_releases else None),
                ("Since",    str(dg_first_year) if dg_first_year else None),
                ("Styles",   "  ·  ".join(dg_styles[:5]) if dg_styles else None),
                ("Labels",   ", ".join(dg_labels[:5]) if dg_labels else None),
            ])
            if dg_url:
                st.markdown(f"[Open on Discogs ↗]({dg_url})")

    yt_subs = enriched.get("yt_subscribers")
    yt_views= enriched.get("yt_views")
    yt_br   = enriched.get("yt_boiler_room")
    yt_ra   = enriched.get("yt_ra_exchange")
    if yt_subs or yt_br or yt_ra:
        with st.container(border=True):
            st.markdown("**YouTube**")
            _kv_grid([
                ("Subscribers", f"{yt_subs:,}" if yt_subs else None),
                ("Total views", f"{yt_views:,}" if yt_views else None),
                ("Boiler Room", "✓ Detected" if yt_br else None),
                ("RA Exchange", "✓ Detected" if yt_ra else None),
            ])

    mc_api_followers = enriched.get("mc_followers")
    mc_api_listens   = enriched.get("mc_listen_count")
    mc_api_tracks    = enriched.get("mc_track_count")
    mc_count = enriched.get("mixcloud_appearances") or 0
    mc_shows = list(dict.fromkeys(enriched.get("mixcloud_shows") or []))
    if mc_api_followers or mc_api_tracks:
        with st.container(border=True):
            st.markdown("**Mixcloud**")
            _kv_grid([
                ("Followers",     f"{mc_api_followers:,}" if mc_api_followers else None),
                ("Total listens", f"{mc_api_listens:,}"   if mc_api_listens else None),
                ("Mixes",         str(mc_api_tracks) if mc_api_tracks else None),
                ("Episode count", str(mc_count) if mc_count else None),
                ("Shows",         ", ".join(mc_shows[:4]) if mc_shows else None),
            ])
    elif mc_count or mc_shows:
        with st.container(border=True):
            st.markdown("**Mixcloud**")
            _kv_grid([
                ("Appearances", str(mc_count)),
                ("Shows",       ", ".join(mc_shows[:6]) if mc_shows else None),
            ])

    achieved = {k: v for k, v in milestones.items() if v}
    if achieved:
        with st.container(border=True):
            st.markdown("**Milestones**")
            _kv_grid([(
                _MILESTONE_LABELS.get(k, k.replace("_", " ").title()), str(v)
            ) for k, v in achieved.items()])

    feedback = enriched.get("lofi_feedback_history") or []
    if feedback:
        st.caption("LOFI feedback: " + "  |  ".join(
            f['decision'] + (f' — {f["note"]}' if f.get('note') else '')
            for f in feedback[-3:]
        ))


if __name__ == "__main__":
    main()
