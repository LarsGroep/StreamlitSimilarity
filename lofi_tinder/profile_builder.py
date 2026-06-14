"""
Generates natural-language artist profiles via LLM.
Caches results to profiles/artist_profiles.jsonl — never regenerates if cached.

Backend selection (set in .env):
    PROFILE_BACKEND=claude    (default) — Claude Haiku via Anthropic API
    PROFILE_BACKEND=ollama              — local Ollama (free, no API key)
    OLLAMA_MODEL=llama3.2               — which Ollama model to use (default: llama3.2)
    OLLAMA_BASE_URL=http://localhost:11434
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ArtistInput, ArtistProfile

_PROFILES_FILE = Path(__file__).parent.parent / "profiles" / "artist_profiles.jsonl"

_SYSTEM = (
    "You are a talent scout analyst for LOFI, an underground electronic music club in Amsterdam. "
    "Write exactly 2-3 sentences: a sharp opinion on this artist's fit for LOFI "
    "(underground, tech-house and related genres, quality over commercial appeal) "
    "and a clear booking recommendation (book now / watch / pass). "
    "Be direct and opinionated. Output only the sentences, nothing else."
)


# ---------------------------------------------------------------------------
# Prompt builder — uses full enriched data
# ---------------------------------------------------------------------------

def _build_prompt(artist: ArtistInput) -> str:
    e = artist.enriched
    lines = [f"Artist: {artist.name}"]

    # --- Streaming / audience ---
    gh = e.get("growth_history") or {}
    listeners = gh.get("current_listeners") or e.get("spotify_followers")
    if listeners:
        growth_total = gh.get("listener_growth_pct_total")
        days = gh.get("days_tracked", 0)
        if growth_total is not None and days > 0:
            lines.append(
                f"Last.fm listeners: {listeners:,} "
                f"({growth_total:+.1f}% over {days}d tracked)"
            )
        else:
            lines.append(f"Last.fm listeners: {listeners:,}")
    if e.get("pf_fans"):
        lines.append(f"Partyflock fans: {e['pf_fans']:,}")

    # --- Momentum ---
    lines.append(f"Momentum score: {e.get('momentum_score', 0):.0f}/100")

    # --- Booking history ---
    bs = e.get("booking_stats") or {}
    total_bookings = bs.get("total", 0)
    recent = bs.get("recent_12m", 0)
    velocity = bs.get("booking_velocity")
    if total_bookings:
        vel_str = ""
        if velocity is not None:
            vel_str = f", {velocity:.1f}x vs prev year" if velocity != 1.0 else ""
        lines.append(f"Booking history: {total_bookings} total, {recent} in last 12m{vel_str}")

    # --- Festival history ---
    festivals = e.get("festival_history") or []
    if festivals:
        lines.append(f"Festival history: {', '.join(festivals[:6])}")

    # --- Label affiliations ---
    bp_releases = e.get("beatport_releases")
    bp_labels   = e.get("beatport_labels") or []
    bp_tier     = e.get("beatport_label_tier")
    bp_latest   = e.get("beatport_latest_release")
    if bp_releases and bp_labels:
        tier_str   = f" [tier {bp_tier}]" if bp_tier else ""
        latest_str = f", latest {bp_latest}" if bp_latest else ""
        lines.append(f"Beatport: {bp_releases} releases on {', '.join(bp_labels[:3])}{tier_str}{latest_str}")

    # --- Agency (gap noted) ---
    if e.get("agency"):
        lines.append(f"Agency: {e['agency']} [{e.get('agency_tier', '')}]")
    else:
        lines.append("Agency: unknown (not yet scraped)")

    # --- Geographic growth ---
    geo_spread  = bs.get("geo_spread", 0)
    countries   = (bs.get("countries") or e.get("geo_countries") or [])[:6]
    nl_ratio    = bs.get("nl_ratio") or e.get("nl_ratio")
    if geo_spread:
        nl_str = f", {nl_ratio:.0%} NL" if nl_ratio is not None else ""
        lines.append(f"Geographic reach: {geo_spread} countries — {', '.join(countries)}{nl_str}")

    # --- Media / validation ---
    mc = e.get("mixcloud_appearances", 0)
    mc_shows = e.get("mixcloud_shows") or []
    if mc:
        show_str = f" ({', '.join(list(dict.fromkeys(mc_shows))[:3])})" if mc_shows else ""
        lines.append(f"Mixcloud show features: {mc}{show_str}")
    ra_events = e.get("ra_genre_events", 0)
    if ra_events:
        lines.append(f"RA genre events: {ra_events}")

    # --- Genre / sound identity ---
    tags = list(dict.fromkeys(
        (e.get("lastfm_tags") or []) + (e.get("ra_genres") or []) + (e.get("spotify_genres") or [])
    ))
    if tags:
        lines.append(f"Genre tags: {', '.join(tags[:7])}")

    # --- Similar artists ---
    similar = list(dict.fromkeys(e.get("lastfm_similar") or []))
    if similar:
        lines.append(f"Similar to: {', '.join(similar[:5])}")

    # NOTE: lofi_booked and lofi_appearance_count intentionally excluded from the prompt.
    # Including them leaks the label into the text embedding — the LLM would write
    # a different (more positive) profile for "LOFI appearances: 3" than for an unknown
    # artist with identical career stats, biasing the 384-dim semantic embedding.

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _generate_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _generate_ollama(prompt: str) -> str:
    import urllib.request
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = os.environ.get("OLLAMA_MODEL", "llama3.2")
    payload  = json.dumps({
        "model": model,
        "prompt": f"{_SYSTEM}\n\n{prompt}",
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        return data.get("response", "").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}\n"
            f"Is Ollama running? Start with: ollama serve\n"
            f"Then pull the model: ollama pull {model}"
        ) from exc


def _generate_template(artist: "ArtistInput") -> str:
    """
    Rule-based profile — fills a structured template from enriched stats.
    No API calls, instant, works on Streamlit Cloud with zero cost.
    Produces enough descriptive content for a meaningful sentence embedding.
    """
    e = artist.enriched or {}
    bs = e.get("booking_stats") or {}
    gh = e.get("growth_history") or {}

    total     = bs.get("total") or 0
    recent_12 = bs.get("recent_12m") or 0
    vel       = bs.get("booking_velocity") or 1.0
    bp_tier   = e.get("beatport_label_tier")
    bp_labels = e.get("beatport_labels") or []
    festivals = e.get("festival_history") or []
    nl_events = int(bs.get("nl_events") or 0)
    geo       = bs.get("geo_spread") or 0
    listeners = gh.get("current_listeners") or e.get("spotify_followers")
    growth    = gh.get("listener_growth_pct_total")

    tags = list(dict.fromkeys(
        (e.get("lastfm_tags") or []) +
        (e.get("ra_genres") or []) +
        (e.get("spotify_genres") or [])
    ))[:4]
    similar = list(dict.fromkeys(
        (e.get("lastfm_similar") or []) +
        (e.get("spotify_related") or [])
    ))[:5]

    if total >= 400 or (total >= 200 and bp_tier in ("A+", "A")):
        stage = "established"
    elif total >= 80 or (total >= 40 and vel >= 1.3):
        stage = "rising"
    elif total >= 15:
        stage = "emerging"
    else:
        stage = "underground"

    genre_str = f"{', '.join(tags)} " if tags else ""
    parts: list[str] = [f"{artist.name} is a {stage} {genre_str}artist."]

    if total:
        vel_note = ""
        if vel >= 1.3:
            vel_note = f", accelerating at {vel:.1f}×"
        elif vel < 0.8:
            vel_note = f", slowing at {vel:.1f}×"
        parts.append(f"{total} career bookings, {recent_12} in the last 12 months{vel_note}.")

    if listeners:
        growth_str = f" ({growth:+.1f}% growth)" if growth else ""
        parts.append(f"{listeners:,} Last.fm listeners{growth_str}.")

    if nl_events >= 6:
        parts.append(f"Saturated in the Netherlands ({nl_events} NL bookings/yr).")
    elif nl_events >= 2:
        parts.append(f"Active in NL ({nl_events} bookings/yr) — accessible but competitive.")
    elif geo > 3:
        parts.append(f"International reach across {geo} countries, not yet established in NL.")
    else:
        parts.append("No significant NL presence yet — potential first-booking opportunity.")

    if bp_labels and bp_tier:
        parts.append(f"Releases on {', '.join(bp_labels[:3])} (label tier {bp_tier}).")

    if festivals:
        parts.append(f"Festival credits: {', '.join(festivals[:4])}.")

    if similar:
        parts.append(f"Similar to {', '.join(similar)}.")

    return " ".join(parts)


def _generate(prompt: str) -> str:
    backend = os.environ.get("PROFILE_BACKEND", "claude").lower()
    if backend == "ollama":
        return _generate_ollama(prompt)
    return _generate_claude(prompt)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, ArtistProfile]:
    cache: dict[str, ArtistProfile] = {}
    if not _PROFILES_FILE.exists():
        return cache
    for line in _PROFILES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                p = ArtistProfile(**data)
                cache[p.artist_id] = p
            except Exception:
                pass
    return cache


def _append_cache(profile: ArtistProfile) -> None:
    _PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROFILES_FILE, "a", encoding="utf-8") as f:
        f.write(profile.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_profile_template(artist: ArtistInput, cache: dict[str, ArtistProfile] | None = None) -> ArtistProfile:
    """Rule-based profile — instant, no API cost. Used for new-artist discovery."""
    if cache is None:
        cache = _load_cache()
    if artist.artist_id in cache:
        return cache[artist.artist_id]
    profile_text = _generate_template(artist)
    profile = ArtistProfile(
        artist_id=artist.artist_id,
        name=artist.name,
        profile_text=profile_text,
        embedding=[],
        cosine_dist_to_centroid=1.0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    _append_cache(profile)
    cache[artist.artist_id] = profile
    return profile


def generate_profile(artist: ArtistInput, cache: dict[str, ArtistProfile] | None = None) -> ArtistProfile:
    if cache is None:
        cache = _load_cache()
    if artist.artist_id in cache:
        return cache[artist.artist_id]

    backend = os.environ.get("PROFILE_BACKEND", "claude").lower()
    if backend == "template":
        return generate_profile_template(artist, cache)

    prompt = _build_prompt(artist)
    profile_text = _generate(prompt)

    profile = ArtistProfile(
        artist_id=artist.artist_id,
        name=artist.name,
        profile_text=profile_text,
        embedding=[],
        cosine_dist_to_centroid=1.0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    _append_cache(profile)
    cache[artist.artist_id] = profile
    return profile


def generate_profiles_batch(
    artists: list[ArtistInput],
    verbose: bool = True,
    max_workers: int = 10,
) -> dict[str, ArtistProfile]:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = _load_cache()
    backend = os.environ.get("PROFILE_BACKEND", "claude").lower()
    cached_count = sum(1 for a in artists if a.artist_id in cache)
    to_generate = [a for a in artists if a.artist_id not in cache]

    if verbose:
        print(f"Profile cache: {cached_count}/{len(artists)} already generated (backend: {backend})")
        if to_generate:
            actual_workers = 1 if backend == "ollama" else max_workers
            print(f"Generating {len(to_generate)} profiles with {actual_workers} worker(s) (backend: {backend})...")

    results: dict[str, ArtistProfile] = {a.artist_id: cache[a.artist_id] for a in artists if a.artist_id in cache}
    lock = threading.Lock()
    in_flight: set[str] = set()   # prevent duplicate API calls under parallelism
    done_count = 0

    def _generate_one(artist: ArtistInput):
        nonlocal done_count
        with lock:
            if artist.artist_id in results or artist.artist_id in in_flight:
                return  # already done or another worker has it
            in_flight.add(artist.artist_id)
        if backend == "template":
            profile_text = _generate_template(artist)
        else:
            prompt = _build_prompt(artist)
            profile_text = _generate(prompt)
        profile = ArtistProfile(
            artist_id=artist.artist_id,
            name=artist.name,
            profile_text=profile_text,
            embedding=[],
            cosine_dist_to_centroid=1.0,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        with lock:
            _append_cache(profile)
            results[artist.artist_id] = profile
            in_flight.discard(artist.artist_id)
            done_count += 1
            if verbose:
                print(f"  [{done_count}/{len(to_generate)}] {artist.name}", flush=True)
        return profile

    if to_generate:
        # Ollama is single-threaded locally — no benefit from parallelism
        workers = 1 if backend == "ollama" else max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_generate_one, a): a for a in to_generate}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    artist = futures[future]
                    print(f"  ERROR generating {artist.name}: {exc}", flush=True)

    return results
