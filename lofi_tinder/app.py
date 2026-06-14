"""
LOFI Tinder — Streamlit artist discovery UI.

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

# Allow running directly via `streamlit run lofi_tinder/app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# On Streamlit Cloud, inject st.secrets into os.environ so downstream code finds them
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
from lofi_tinder.ranker import get_swiped_ids, load_swipes, rank_candidates
from lofi_tinder.schemas import ArtistProfile, SwipeRecord

_NEGATIVE_DECISIONS = {"no", "commercial", "wrong_genre", "saturated_nl", "not_ready"}

_DATA_DIR = Path(__file__).parent.parent / "data"
_PROFILES_FILE = Path(__file__).parent.parent / "profiles" / "artist_profiles.jsonl"
_SWIPES_FILE = _DATA_DIR / "swipes.jsonl"
_CENTROID_UPDATE_EVERY = 20   # YES swipes before centroid refresh

st.set_page_config(page_title="LOFI Tinder", page_icon="🎛", layout="wide")


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


def _save_swipe(swipe: SwipeRecord) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SWIPES_FILE, "a", encoding="utf-8") as f:
        f.write(swipe.model_dump_json() + "\n")


def _update_centroid_from_swipes(swipes: list[SwipeRecord], profiles: dict[str, ArtistProfile]) -> None:
    yes_swipes = [
        s for s in swipes
        if s.decision == "yes" and s.artist_id in profiles and profiles[s.artist_id].embedding
    ]
    if not yes_swipes:
        return

    # Text centroid (384-dim) — used as display fallback
    centroid = compute_centroid([profiles[s.artist_id].embedding for s in yes_swipes])
    save_centroid(centroid)

    # Feature centroid (15-dim) — primary ranking signal
    emap = _load_enriched_map()
    feature_vecs = []
    for s in yes_swipes:
        enriched = emap.get(s.artist_id) or {}
        if not enriched:
            # fall back to candidates map for discovered artists
            cdata = _load_candidates_map().get(s.artist_id, {})
            enriched = cdata.get("enriched", {})
        if enriched:
            feature_vecs.append(extract_feature_vector(enriched))
    if feature_vecs:
        save_feature_centroid(compute_feature_centroid(feature_vecs))

    st.cache_data.clear()


def _count_yes(swipes: list[SwipeRecord]) -> int:
    return sum(1 for s in swipes if s.decision == "yes")


def _count_neg(swipes: list[SwipeRecord]) -> int:
    return sum(1 for s in swipes if s.decision in _NEGATIVE_DECISIONS)


def _handle_swipe(
    artist: ArtistProfile,
    decision: str,
    mab: LinUCB,
    mab_scores: dict,
    enriched_map: dict,
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
        enriched = enriched_map.get(artist.artist_id) or {}
        if enriched:
            import numpy as np
            fvec = extract_feature_vector(enriched)
            mab.update(fvec.astype("float64"), reward)
            mab.save()
    st.session_state["queue_idx"] = st.session_state.get("queue_idx", 0) + 1
    st.rerun()


def _show_batch_end(
    swipes: list,
    profiles: dict,
    swiped_ids: set,
    mab,
    mab_scores: dict,
) -> None:
    """Shown when the current batch of 20 is exhausted. Triggers discovery of next 20."""
    from lofi_tinder.discover import discover_new_batch

    # Batch summary — last 20 non-pre-seeded swipes
    last_batch   = swipes[-20:] if len(swipes) >= 20 else swipes
    yes_names    = [s.name for s in last_batch if s.decision == "yes"]
    monitor_names = [s.name for s in last_batch if s.decision == "monitor"]
    no_names     = [s.name for s in last_batch if s.decision in _NEGATIVE_DECISIONS]
    skip_names   = [s.name for s in last_batch if s.decision == "skip"]
    # Seed discovery from YES + MONITOR (both are positive signals)
    seed_names   = yes_names + monitor_names

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

    # Show breakdown of negative reasons if any
    if no_names:
        reason_counts: dict[str, int] = {}
        for s in last_batch:
            if s.decision in _NEGATIVE_DECISIONS:
                reason_counts[s.decision] = reason_counts.get(s.decision, 0) + 1
        if reason_counts:
            st.caption("Rejections: " + "  ·  ".join(
                f"{k.replace('_', ' ')} ({v})" for k, v in reason_counts.items()
            ))

    st.divider()
    st.markdown(
        "LOFI Feel updated from swipes. "
        "Click below to discover 20 new artists via Last.fm similar-artist chains "
        "seeded from your YES and MONITOR choices."
    )

    if not seed_names:
        st.warning("No YES or MONITOR swipes in last batch — need at least one positive to seed discovery.")
        if st.button("Reset queue (re-rank existing candidates)"):
            st.session_state.pop("queue", None)
            st.session_state["queue_stale"] = True
            st.rerun()
        return

    if st.button("Find next 20 artists", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Discovering new artists...")
        status_text  = st.empty()

        def _progress(done: int, total: int, name: str):
            pct = min(done / max(total, 1), 1.0)
            progress_bar.progress(pct, text=f"Profiling {name}…" if name != "done" else "Done!")
            status_text.caption(f"{done}/{total} artists processed")

        new_ids = discover_new_batch(
            yes_names  = seed_names,
            swiped_ids = swiped_ids,
            profiles   = profiles,
            n          = 20,
            progress_cb= _progress,
        )

        progress_bar.empty()
        status_text.empty()

        if new_ids:
            st.success(f"Found {len(new_ids)} new artists. Loading next batch…")
            # Clear cache + queue so new profiles load
            st.cache_data.clear()
            st.session_state.pop("queue", None)
            st.session_state["queue_idx"]   = 0
            st.session_state["queue_stale"] = False
            time.sleep(0.5)
            st.rerun()
        else:
            st.warning(
                "No new artists found via Last.fm similar-artist chains. "
                "Try running `python run.py --candidates` to replenish from the full database."
            )
            if st.button("Reset queue with existing candidates"):
                st.session_state.pop("queue", None)
                st.session_state["queue_stale"] = True
                st.rerun()


def main() -> None:
    st.title("🎛 LOFI Artist Tinder")

    profiles = _load_profiles()
    if not profiles:
        st.error("No artist profiles found. Run: `python run.py --seed && python run.py --candidates`")
        return

    swipes = load_swipes()
    swiped_ids = get_swiped_ids(swipes)
    yes_count  = _count_yes(swipes)
    neg_count  = _count_neg(swipes)

    # Enriched map — used for ranking, MAB feature vectors, and card display
    emap = _load_enriched_map()

    # MAB: score using 14-dim feature vectors (tractable with few swipes)
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

    # Get ranked queue (primary signal: 14-dim structured feature similarity)
    if "queue" not in st.session_state or st.session_state.get("queue_stale"):
        queue = rank_candidates(profiles, swiped_ids, emap, mab_scores, limit=20)
        st.session_state["queue"] = queue
        st.session_state["queue_idx"] = 0
        st.session_state["queue_stale"] = False

    queue: list[ArtistProfile] = st.session_state["queue"]
    idx: int = st.session_state["queue_idx"]

    # Session stats
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    col_stat1.metric("Total swiped", len(swiped_ids))
    col_stat2.metric("YES", yes_count)
    col_stat3.metric("Monitor", sum(1 for s in swipes if s.decision == "monitor"))
    col_stat4.metric("No", neg_count)
    col_stat5.metric("Remaining", max(0, len(queue) - idx))

    # Centroid update progress
    next_update_in = _CENTROID_UPDATE_EVERY - (yes_count % _CENTROID_UPDATE_EVERY)
    if yes_count > 0 and next_update_in == _CENTROID_UPDATE_EVERY:
        st.success(f"Taste profile updated! ({yes_count} YES swipes total)")
    else:
        st.info(f"Centroid update in {next_update_in} more YES swipe(s)")

    # Check if centroid update is due
    if yes_count > 0 and yes_count % _CENTROID_UPDATE_EVERY == 0:
        if st.session_state.get("last_centroid_update") != yes_count:
            _update_centroid_from_swipes(swipes, profiles)
            mab.save()
            st.session_state["last_centroid_update"] = yes_count
            st.session_state["queue_stale"] = True

    # Show current card
    if idx >= len(queue):
        _show_batch_end(swipes, profiles, swiped_ids, mab, mab_scores)
        return

    artist = queue[idx]
    dist   = artist.cosine_dist_to_centroid

    st.divider()

    # --- HTML header: photo + name only (scores rendered inside _show_stats) ---
    _enr    = emap.get(artist.artist_id) or {}
    img_url = _enr.get("image_url") or _fetch_spotify_image(artist.name)
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

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:20px;padding:8px 0 4px 0">
      {img_html}
      <div style="flex:1;min-width:0">
        <h2 style="margin:0;font-size:2em;font-weight:700;line-height:1.1">{artist.name}</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Full-width profile card (scores + detail) ---
    _show_stats(artist.artist_id, artist.profile_text, dist,
                nearest_cluster=getattr(artist, "nearest_cluster", "unknown"))

    st.divider()

    # --- Swipe buttons: positives on top, negatives below ---
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
        if st.button("No fit", use_container_width=True, type="secondary", key="swipe_no"):
            _handle_swipe(artist, "no", mab, mab_scores, emap)
    with r2:
        if st.button("Too commercial", use_container_width=True, type="secondary", key="swipe_commercial"):
            _handle_swipe(artist, "commercial", mab, mab_scores, emap)
    with r3:
        if st.button("Wrong genre", use_container_width=True, type="secondary", key="swipe_genre"):
            _handle_swipe(artist, "wrong_genre", mab, mab_scores, emap)
    with r4:
        if st.button("Saturated NL", use_container_width=True, type="secondary", key="swipe_saturated"):
            _handle_swipe(artist, "saturated_nl", mab, mab_scores, emap)
    with r5:
        if st.button("Not ready yet", use_container_width=True, type="secondary", key="swipe_notready"):
            _handle_swipe(artist, "not_ready", mab, mab_scores, emap)


@st.cache_data(ttl=300)
def _load_candidates_map() -> dict[str, dict]:
    candidates_file = _DATA_DIR / "candidates.jsonl"
    result = {}
    if not candidates_file.exists():
        return result
    for line in candidates_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                result[data.get("artist_id", "")] = data
            except Exception:
                pass
    return result


@st.cache_data(ttl=300)
def _load_enriched_map() -> dict[str, dict]:
    enriched_file = Path(__file__).parent.parent / "scraper_data" / "artist_enriched.jsonl"
    result = {}
    if not enriched_file.exists():
        return result
    for line in enriched_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                aid = data.get("artist_id", "")
                result[aid] = data
                # Also index by any old slugs so merged artists are still found
                for old_id in data.get("old_artist_ids") or []:
                    result[old_id] = data
            except Exception:
                pass
    return result


@st.cache_data(ttl=3600)
def _fetch_spotify_image(artist_name: str) -> str | None:
    import base64
    import re
    import urllib.request
    import urllib.parse
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None

    # Strip country/region suffixes like "(UK)", "(NL)", "(US)", "(DE)" etc.
    clean_name = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", artist_name).strip()

    try:
        creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
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

        def _search(name: str) -> str | None:
            q = urllib.parse.quote(name)
            req2 = urllib.request.Request(
                f"https://api.spotify.com/v1/search?q={q}&type=artist&limit=3",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req2, timeout=5) as r:
                data = json.loads(r.read())
            items = (data.get("artists") or {}).get("items") or []
            if items:
                images = items[0].get("images") or []
                if images:
                    return images[min(1, len(images) - 1)].get("url")
            return None

        # Try clean name first, fall back to original if different
        result = _search(clean_name)
        if not result and clean_name != artist_name:
            result = _search(artist_name)
        return result
    except Exception:
        pass
    return None


@st.cache_data(ttl=600)
def _load_label_artists_map() -> dict[str, list[str]]:
    v2 = Path(__file__).parent.parent.parent / "v2-scraper" / "scraper"
    path = v2 / "BeatportLabelArtistItem.jsonl"
    by_label: dict[str, list[str]] = {}
    if not path.exists():
        return by_label
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            label  = (row.get("label_name") or "").lower()
            artist = row.get("artist_name") or ""
            if label and artist:
                by_label.setdefault(label, [])
                if artist not in by_label[label]:
                    by_label[label].append(artist)
        except Exception:
            pass
    return by_label



def _kv_grid(items: list[tuple[str, str]]) -> None:
    """Render a compact spreadsheet-style row: label on top, value below. Only non-empty items."""
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
        import pandas as pd
        import altair as alt
        # Accept either "listeners" key (Last.fm) or pre-mapped dicts from SC
        rows = [{"date": s["date"], y_col: s.get("listeners") or s.get(y_col)}
                for s in snapshots if s.get("date") and (s.get("listeners") or s.get(y_col))]
        rows = [r for r in rows if r[y_col] is not None]
        if len(rows) < 2:
            return
        df = (pd.DataFrame(rows)
              .drop_duplicates("date")
              .assign(date=lambda x: pd.to_datetime(x["date"]))
              .sort_values("date"))
        chart = (
            alt.Chart(df)
            .mark_area(
                line={"color": color, "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear", x1=0, x2=0, y1=1, y2=0,
                    stops=[
                        alt.GradientStop(color=color, offset=1),
                        alt.GradientStop(color="transparent", offset=0),
                    ],
                ),
            )
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)),
                y=alt.Y(f"{y_col}:Q", title=None, axis=alt.Axis(format=",.0f")),
            )
            .properties(height=150)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass


def _score_breakdown(enriched: dict, cosine_dist: float) -> dict[str, str]:
    """Return one-line explanation per score showing inputs and component weights."""
    import math

    bs         = enriched.get("booking_stats") or {}
    gh         = enriched.get("growth_history") or {}
    vel        = bs.get("booking_velocity") or 0.0
    geo        = bs.get("geo_spread") or 0
    nl_events  = int(round(bs.get("nl_events") or 0))
    bp_tier    = enriched.get("beatport_label_tier")
    festivals  = enriched.get("festival_history") or []
    mc         = enriched.get("mixcloud_appearances") or 0
    milestones = enriched.get("milestones") or {}
    total      = bs.get("total") or 0
    recent_12  = bs.get("recent_12m") or 0
    ra_ev      = enriched.get("ra_genre_events") or 0
    listeners  = gh.get("current_listeners") or enriched.get("spotify_followers") or 0
    pf_fans    = enriched.get("pf_fans") or 0

    # ── Sound Fit ──
    sf = max(0, min(100, int((1 - cosine_dist) * 100)))
    sound_fit_txt = (
        f"{sf}/100 — cosine similarity to nearest LOFI feature centroid (core or emerging). "
        f"Inputs: booking velocity, label tier, geo spread, listener scale, NL ratio, Beatport activity."
    )

    # ── Heat ──
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

    heat = min(100, book_pts + vel_pts + geo_pts + aud_pts)
    vel_label = "growing" if vel >= 1.2 else "stable" if vel >= 0.9 else "declining"
    heat_txt = (
        f"{heat}/100 = bookings {book_pts}pts ({recent_12}/yr in last 12m) + "
        f"velocity {vel_pts}pts ({vel:.1f}×, {vel_label}) + "
        f"geo reach {geo_pts}pts ({geo} countries) + "
        f"audience {aud_pts}pts ({aud:,} listeners)"
    )

    # ── Window ──
    nl_sat_pts = max(0, 60 - nl_events * 10)
    vel_rising = min(40, int(max(0, vel - 1.0) * 40))
    window     = min(100, nl_sat_pts + vel_rising)
    if nl_events >= 6:
        nl_window_note = f"0pts — {nl_events} NL bookings/yr (saturated)"
    elif nl_events >= 3:
        nl_window_note = f"{nl_sat_pts}pts — {nl_events} NL bookings/yr (active, limited window)"
    elif nl_events >= 1:
        nl_window_note = f"{nl_sat_pts}pts — {nl_events} NL bookings/yr (low NL, good window)"
    else:
        nl_window_note = f"{nl_sat_pts}pts — no NL presence yet (max window)"
    window_txt = (
        f"{window}/100 = NL availability {nl_window_note} + "
        f"trajectory {vel_rising}pts (vel={vel:.1f}×; rising above 1.0× = actively growing). "
        f"High = early opportunity in NL. Low = already established or saturated."
    )

    # ── Track Record ──
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

    hs_names = [_MILESTONE_LABELS.get(k, k) for k in _HIGH_SIGNAL_MILESTONES if milestones.get(k)]
    missing = []
    if not bp_tier:
        missing.append("no label tier")
    if not mc:
        missing.append("no Mixcloud data")
    missing_note = f" (missing: {', '.join(missing)})" if missing else ""
    track_record_txt = (
        f"{track_record}/100 = festivals {fest_pts}pts ({nf} known) + "
        f"label prestige {bp_pts}pts ({bp_tier or 'unknown tier'}) + "
        f"career depth {depth_pts}pts ({total} total bookings) + "
        f"industry {ind_pts}pts (RA={ra_pts}, Mixcloud={mc_pts}, milestones={ms_pts})"
        + (f" [{', '.join(hs_names[:3])}]" if hs_names else "")
        + missing_note
    )

    return {
        "Sound Fit":    sound_fit_txt,
        "Heat":         heat_txt,
        "Window":       window_txt,
        "Track Record": track_record_txt,
    }


_MILESTONE_LABELS = {
    "first_ibiza":           "First Ibiza booking",
    "first_circoloco":       "First Circoloco",
    "first_music_on":        "First Music On",
    "first_ants":            "First ANTS",
    "first_piv_release":     "First PIV release",
    "first_beatport_top10":  "First Beatport Top 10",
    "first_beatport_no1":    "First Beatport #1",
    "first_festival":        "First festival",
    "first_boiler_room":     "First Boiler Room",
    "first_ra_podcast":      "First RA Podcast",
    "first_bbc_r1":          "First BBC Radio 1",
    "first_headline_500":    "First headline 500+",
    "first_headline_1000":   "First headline 1,000+",
    "first_headline_2000":   "First headline 2,000+",
    "first_headline_5000":   "First headline 5,000+",
    "first_tier_a_support":  "First Tier A support",
    "first_tier_a_b2b":      "First Tier A B2B",
    "first_extended_set":    "First extended set",
    "first_anl":             "First All Night Long",
    "first_adl":             "First All Day Long",
    "first_major_residency": "First major residency",
    "first_multi_city_tour": "First multi-city tour",
}

_NOTABLE_PENDING = {
    "first_ibiza", "first_circoloco", "first_music_on", "first_ants",
    "first_beatport_top10", "first_boiler_room", "first_ra_podcast", "first_bbc_r1",
}

# Milestones that genuinely matter to a talent buyer (high signal, not noise)
_HIGH_SIGNAL_MILESTONES = {
    "first_circoloco", "first_music_on", "first_ants",
    "first_beatport_top10", "first_beatport_no1",
    "first_boiler_room", "first_ra_podcast", "first_bbc_r1",
    "first_headline_1000", "first_headline_2000", "first_headline_5000",
    "first_tier_a_b2b",
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

    # ── Sound Fit ────────────────────────────────────────────────────────────
    # Cosine similarity to nearest LOFI centroid → 0–100.
    sound_fit = max(0, min(100, int((1 - cosine_dist) * 100)))

    # ── Heat ─────────────────────────────────────────────────────────────────
    # "Is this artist actively in-demand right now?"
    # Captures both SCALE of activity and DIRECTION — an artist doing 100 shows
    # at 0.8× still has far more heat than one doing 3 shows at 2×.

    # Recent booking volume (0–40pts, log scale so scale matters but doesn't dominate)
    if recent_12 == 0:     book_pts = 0
    elif recent_12 <= 2:   book_pts = 8
    elif recent_12 <= 5:   book_pts = 16
    elif recent_12 <= 10:  book_pts = 22
    elif recent_12 <= 20:  book_pts = 28
    elif recent_12 <= 50:  book_pts = 34
    else:                  book_pts = 40

    # Velocity direction bonus (0–20pts); 1.0× = stable baseline = 11pts
    if not vel:            vel_pts = 8    # no data
    elif vel >= 2.0:       vel_pts = 20
    elif vel >= 1.5:       vel_pts = 17
    elif vel >= 1.2:       vel_pts = 14
    elif vel >= 1.0:       vel_pts = 11
    elif vel >= 0.7:       vel_pts = 7
    elif vel >= 0.4:       vel_pts = 3
    else:                  vel_pts = 0

    # International reach (0–20pts)
    if geo == 0:           geo_pts = 0
    elif geo == 1:         geo_pts = 4
    elif geo <= 3:         geo_pts = 8
    elif geo <= 7:         geo_pts = 12
    elif geo <= 14:        geo_pts = 16
    else:                  geo_pts = 20

    # Audience scale (0–20pts) — listeners or Spotify followers, log scale
    aud = max(listeners, pf_fans * 20)   # rough pf_fans → listener-scale normalization
    if aud == 0:           aud_pts = 0
    elif aud < 1_000:      aud_pts = 3
    elif aud < 10_000:     aud_pts = 7
    elif aud < 50_000:     aud_pts = 11
    elif aud < 200_000:    aud_pts = 15
    elif aud < 1_000_000:  aud_pts = 18
    else:                  aud_pts = 20

    heat = min(100, book_pts + vel_pts + geo_pts + aud_pts)

    # ── Window ───────────────────────────────────────────────────────────────
    # "Is there a booking opportunity for LOFI?" — label-free.
    # High = NL market not yet saturated AND trajectory is rising.
    nl_sat_pts = max(0, 60 - nl_events * 10)
    vel_rising = min(40, int(max(0, vel - 1.0) * 40))
    window     = min(100, nl_sat_pts + vel_rising)

    # ── Track Record ─────────────────────────────────────────────────────────
    # "How validated is this artist by the industry?"
    # Built from what we actually have: festivals, label tier, career depth, RA/Mixcloud/milestones.

    # Festival footprint (0–30pts, log scale)
    nf = len(festivals)
    if nf == 0:    fest_pts = 0
    elif nf == 1:  fest_pts = 5
    elif nf <= 3:  fest_pts = 10
    elif nf <= 7:  fest_pts = 17
    elif nf <= 14: fest_pts = 23
    elif nf <= 20: fest_pts = 27
    else:          fest_pts = 30

    # Label prestige (0–25pts)
    bp_pts = {"A+": 25, "A": 18, "B": 10}.get(bp_tier or "", 0)

    # Career depth / longevity (0–25pts, log scale)
    if total == 0:    depth_pts = 0
    elif total < 10:  depth_pts = 4
    elif total < 30:  depth_pts = 8
    elif total < 75:  depth_pts = 12
    elif total < 150: depth_pts = 16
    elif total < 300: depth_pts = 20
    elif total < 500: depth_pts = 23
    else:             depth_pts = 25

    # Industry recognition (0–20pts): RA genre events + Mixcloud + high-signal milestones
    ra_pts  = min(8, int(math.log10(max(ra_ev, 1)) / math.log10(201) * 8))
    mc_pts  = min(6, mc)
    ms_pts  = min(10, sum(5 for k in _HIGH_SIGNAL_MILESTONES if milestones.get(k)))
    ind_pts = min(20, ra_pts + mc_pts + ms_pts)

    track_record = min(100, fest_pts + bp_pts + depth_pts + ind_pts)

    # ── Career stage ─────────────────────────────────────────────────────────
    if total >= 400 or (total >= 200 and bp_tier in ("A+", "A")):
        stage, stage_bg = "Established", "#6366f1"
    elif total >= 80 or (total >= 40 and vel >= 1.3):
        stage, stage_bg = "Rising",      "#16a34a"
    elif total >= 15:
        stage, stage_bg = "Emerging",    "#d97706"
    else:
        stage, stage_bg = "Underground", "#475569"

    # ── NL saturation ────────────────────────────────────────────────────────
    if nl_events >= 8:
        nl_label, nl_bg = "Saturated NL",    "#dc2626"
    elif nl_events >= 4:
        nl_label, nl_bg = "Active in NL",    "#d97706"
    elif nl_events >= 1:
        nl_label, nl_bg = "Low NL presence", "#16a34a"
    else:
        nl_label, nl_bg = "Fresh to NL",     "#16a34a"

    return {
        "sound_fit":    sound_fit,
        "heat":         heat,
        "window":       window,
        "track_record": track_record,
        "stage":        stage,
        "stage_bg":     stage_bg,
        "nl_label":     nl_label,
        "nl_bg":        nl_bg,
        "nl_events":    nl_events,
    }


def _show_lofi_feel_matrix(enriched: dict) -> None:
    """Horizontal grouped bar chart: artist vs both LOFI centroids (core + emerging)."""
    import numpy as np
    import altair as alt
    import pandas as pd
    from lofi_tinder.embedder import (
        extract_feature_vector, load_dual_feature_centroids, load_feature_centroid,
    )

    core_centroid, emerging_centroid = load_dual_feature_centroids()
    single_centroid = load_feature_centroid()

    if core_centroid is None and single_centroid is None:
        st.warning("LOFI centroid not built yet. Run: python run.py --seed")
        return

    artist_vec = extract_feature_vector(enriched)

    labels = [
        "Listeners",
        "Listener growth",
        "Momentum",
        "Career bookings",
        "Booking velocity",
        "Recent bookings",
        "Geo spread",
        "NL ratio",
        "Beatport releases",
        "Label tier",
        "Mixcloud",
        "RA credibility",
        "Festival history",
        "Partyflock fans",
    ]

    rows = []
    for i, label in enumerate(labels):
        rows.append({"Feature": label, "Series": "Artist",           "Score": float(artist_vec[i])})
        if core_centroid is not None:
            rows.append({"Feature": label, "Series": "LOFI core",    "Score": float(core_centroid[i])})
        if emerging_centroid is not None:
            rows.append({"Feature": label, "Series": "LOFI emerging","Score": float(emerging_centroid[i])})
        if core_centroid is None and single_centroid is not None:
            rows.append({"Feature": label, "Series": "LOFI avg",     "Score": float(single_centroid[i])})

    series_order = ["Artist", "LOFI core", "LOFI emerging"]
    color_range  = ["#4ade80", "#818cf8", "#34d399"]

    df = pd.DataFrame(rows)
    feature_order = list(reversed(labels))

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Feature:N", sort=feature_order, title=None,
                    axis=alt.Axis(labelLimit=140, labelFontSize=12)),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score (0–1)",
                    axis=alt.Axis(grid=True, gridOpacity=0.3)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=series_order, range=color_range),
                legend=alt.Legend(orient="top", title=None, symbolSize=80),
            ),
            yOffset=alt.YOffset("Series:N", sort=series_order),
            tooltip=["Feature", "Series", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(
            height=430,
            title=alt.TitleParams(
                "LOFI Feel Matrix — artist vs Core (established) and Emerging centroids",
                fontSize=12,
            ),
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Nearest-cluster similarity summary
    def _sim(v, c):
        vn, cn = np.linalg.norm(v), np.linalg.norm(c)
        return float(np.dot(v, c) / (vn * cn)) if vn > 0 and cn > 0 else 0.0

    lines = []
    if core_centroid is not None:
        cs = _sim(artist_vec, core_centroid)
        lines.append(f"Core (established): **{cs:.0%}**")
    if emerging_centroid is not None:
        es = _sim(artist_vec, emerging_centroid)
        lines.append(f"Emerging: **{es:.0%}**")
    if lines:
        nearest = "core" if (core_centroid is not None and emerging_centroid is not None and
                             _sim(artist_vec, core_centroid) >= _sim(artist_vec, emerging_centroid)) else "emerging"
        st.caption("Similarity — " + "  ·  ".join(lines) + f"  ·  Nearest cluster: **{nearest}**")


def _show_stats(artist_id: str, profile_text: str, cosine_dist: float = 1.0,
                nearest_cluster: str = "unknown") -> None:
    emap = _load_enriched_map()
    enriched = emap.get(artist_id)
    if enriched is None:
        cmap = _load_candidates_map()
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

    # ── Score pills ──────────────────────────────────────────────────────────
    def _pill(value, label, bg, wide=False):
        w = "min-width:90px" if wide else "min-width:72px"
        return (
            f'<div style="{w};padding:10px 14px;border-radius:10px;background:{bg};'
            f'color:white;text-align:center;display:inline-block">'
            f'<div style="font-size:1.5em;font-weight:800;line-height:1">{value}</div>'
            f'<div style="font-size:0.68em;margin-top:3px;opacity:0.9;letter-spacing:.5px">'
            f'{label.upper()}</div></div>'
        )
    cluster_pill_bg = {"core": "#4f46e5", "emerging": "#0d9488"}.get(nearest_cluster, "#475569")
    cluster_pill_lbl = {"core": "Core", "emerging": "Emerging"}.get(nearest_cluster, "?")
    pills_html = " ".join([
        _pill(scores["sound_fit"],    "Sound Fit",    "#1d4ed8"),
        _pill(scores["heat"],         "Heat",         "#15803d"),
        _pill(scores["window"],       "Window",       "#7c3aed"),
        _pill(scores["track_record"], "Track Record", "#b45309"),
        _pill(scores["stage"],        "Career Stage", scores["stage_bg"], wide=True),
        _pill(scores["nl_label"],     "NL Status",    scores["nl_bg"],    wide=True),
        _pill(cluster_pill_lbl,       "Cluster",      cluster_pill_bg,    wide=True),
    ])
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 4px 0">{pills_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Score breakdown expander ─────────────────────────────────────────────
    with st.expander("Score breakdown", expanded=False):
        for name, explanation in _score_breakdown(enriched, cosine_dist).items():
            st.markdown(f"**{name}** — {explanation}")

    # ── Genre tags + similar ─────────────────────────────────────────────────
    tags = list(dict.fromkeys(
        (enriched.get("lastfm_tags") or [])
        + (enriched.get("ra_genres") or [])
        + (enriched.get("spotify_genres") or [])
    ))
    similar = list(dict.fromkeys(
        (enriched.get("lastfm_similar") or [])
        + (enriched.get("spotify_related") or [])
    ))
    if tags:
        st.caption("  ·  ".join(f"#{t}" for t in tags[:8]))
    if similar:
        st.caption(f"Similar to: {', '.join(similar[:8])}")

    # ── Key signal line (agency / label / best milestone) ────────────────────
    bp_tier   = enriched.get("beatport_label_tier")
    bp_labels = enriched.get("beatport_labels") or []
    agency    = enriched.get("agency")
    milestones = enriched.get("milestones") or {}
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

    # ── Scout opinion ────────────────────────────────────────────────────────
    if profile_text:
        st.info(profile_text)

    st.divider()

    # =========================================================================
    # PER-SOURCE DATA SECTIONS — only rendered if that source has data
    # =========================================================================

    # ── Partyflock ───────────────────────────────────────────────────────────
    total    = bs.get("total") or 0
    recent12 = bs.get("recent_12m") or 0
    vel      = bs.get("booking_velocity")
    geo      = bs.get("geo_spread") or 0
    nl_ratio = bs.get("nl_ratio")
    pf_fans  = enriched.get("pf_fans") or 0
    all_evs  = bs.get("recent_events") or []
    fh       = enriched.get("festival_history") or []
    countries = bs.get("countries") or []

    if total or pf_fans or all_evs:
        with st.container(border=True):
            st.markdown("**Partyflock**")
            vel_str = None
            if vel:
                arrow = "↗" if vel > 1.1 else "↘" if vel < 0.9 else "→"
                vel_str = f"{vel:.1f}× {arrow}"
            _kv_grid([
                ("Fans",          f"{pf_fans:,}" if pf_fans else None),
                ("Career",        str(total) if total else None),
                ("Last 12m",      str(recent12) if recent12 else None),
                ("Velocity",      vel_str),
                ("NL ratio",      f"{nl_ratio:.0%}" if nl_ratio else None),
                ("Countries",     " · ".join(countries[:12]) if countries else None),
            ])

            # Festivals — list of actual names, not a count
            if fh:
                st.caption("Festivals: " + "  ·  ".join(fh[:20]))

            # Upcoming / recent events
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

    # ── Last.fm ──────────────────────────────────────────────────────────────
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
                ("Listeners",     f"{lfm_listeners:,}" if lfm_listeners else None),
                ("Growth",        f"{lfm_growth:+.1f}%" if lfm_growth is not None else None),
                ("Playcount",     f"{lfm_playcount:,}" if lfm_playcount else None),
                ("Plays/listener",f"{ppl}" if ppl else None),
                ("Tags",          "  ·  ".join(f"#{t}" for t in lfm_tags[:6]) if lfm_tags else None),
                ("Similar",       ", ".join(lfm_similar[:6]) if lfm_similar else None),
            ])

            # Time series chart — main focus
            unique_dates = {(s.get("date") or "")[:10] for s in snaps if s.get("listeners")}
            if len(unique_dates) >= 2:
                _plot_growth(snaps)
            elif snaps:
                st.caption(f"1 snapshot ({list(unique_dates)[0] if unique_dates else '?'}) — run lastfm_enricher weekly to build growth history")

    # ── Spotify ──────────────────────────────────────────────────────────────
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

    # ── Beatport ─────────────────────────────────────────────────────────────
    bp_releases = enriched.get("beatport_releases")
    bp_latest   = enriched.get("beatport_latest_release")

    if bp_releases or bp_labels:
        with st.container(border=True):
            st.markdown("**Beatport**")
            _kv_grid([
                ("Releases",    str(bp_releases) if bp_releases else None),
                ("Label tier",  bp_tier if bp_tier else None),
                ("Latest",      bp_latest if bp_latest else None),
                ("Labels",      ", ".join(bp_labels[:5]) if bp_labels else None),
            ])
            lmap = _load_label_artists_map()
            coartists: list[str] = []
            seen = {(enriched.get("name") or "").lower()}
            for lbl in bp_labels[:3]:
                for a in lmap.get(lbl.lower(), [])[:15]:
                    if a.lower() not in seen:
                        coartists.append(a)
                        seen.add(a.lower())
            if coartists[:8]:
                st.caption("Label mates: " + "  ·  ".join(coartists[:8]))

    # ── SoundCloud ───────────────────────────────────────────────────────────
    sc_followers = enriched.get("sc_followers")
    sc_tracks    = enriched.get("sc_tracks")
    sc_url       = enriched.get("sc_url")
    sc_snaps     = enriched.get("sc_snapshots") or []

    if sc_followers or sc_tracks:
        with st.container(border=True):
            st.markdown("**SoundCloud**")
            sc_unique_dates = {s.get("date", "") for s in sc_snaps if s.get("followers")}
            sc_growth = None
            if len(sc_snaps) >= 2:
                first_f = sc_snaps[0].get("followers") or 0
                last_f  = sc_snaps[-1].get("followers") or 0
                if first_f:
                    sc_growth = f"{(last_f - first_f) / first_f * 100:+.1f}%"
            _kv_grid([
                ("Followers", f"{sc_followers:,}" if sc_followers else None),
                ("Growth",    sc_growth),
                ("Tracks",    str(sc_tracks) if sc_tracks else None),
            ])
            if len(sc_unique_dates) >= 2:
                sc_rows = [{"date": s["date"], "Followers": s["followers"]}
                           for s in sc_snaps if s.get("date") and s.get("followers")]
                _plot_growth(
                    [{"date": r["date"], "listeners": r["Followers"]} for r in sc_rows],
                    y_col="Followers", color="#f97316",
                )
            elif sc_snaps:
                st.caption("1 SC snapshot — run soundcloud_enricher.py weekly to build history")
            if sc_url:
                st.markdown(f"[Open on SoundCloud ↗]({sc_url})")

    # ── Mixcloud ─────────────────────────────────────────────────────────────
    mc_count = enriched.get("mixcloud_appearances") or 0
    mc_shows = list(dict.fromkeys(enriched.get("mixcloud_shows") or []))
    if mc_count:
        with st.container(border=True):
            st.markdown("**Mixcloud**")
            _kv_grid([
                ("Appearances", str(mc_count)),
                ("Shows",       ", ".join(mc_shows[:6]) if mc_shows else None),
            ])

    # ── RA ───────────────────────────────────────────────────────────────────
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

    # ── Milestones ───────────────────────────────────────────────────────────
    achieved = {k: v for k, v in milestones.items() if v}
    if achieved:
        with st.container(border=True):
            st.markdown("**Milestones**")
            items = []
            for k, v in achieved.items():
                label = _MILESTONE_LABELS.get(k, k.replace("_", " ").title())
                items.append((label, str(v)))
            _kv_grid(items)

    # ── LOFI Feel Matrix ─────────────────────────────────────────────────────
    with st.expander("LOFI Feel Matrix — compare against LOFI taste profile", expanded=False):
        _show_lofi_feel_matrix(enriched)

    # ── LOFI history ─────────────────────────────────────────────────────────
    feedback = enriched.get("lofi_feedback_history") or []
    if feedback:
        st.caption("LOFI feedback: " + "  |  ".join(
            f['decision'] + (f' — {f["note"]}' if f.get('note') else '')
            for f in feedback[-3:]
        ))


if __name__ == "__main__":
    main()
