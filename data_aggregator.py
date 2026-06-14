"""
Data aggregator — reads all scraper sources and produces one rich record per artist.

Output: scraper_data/artist_enriched.jsonl

Run before --seed or --candidates:
    python data_aggregator.py

Sources read:
    ra-scraper-master/scraper/lastfm/LastFMSnapshot.jsonl    (growth history)
    ra-scraper-master/scraper/PartyflockEventItem.jsonl      (booking history, geo)
    ra-scraper-master/scraper/PartyflockLineupItem.jsonl     (venue/festival names)
    ra-scraper-master/scraper/FestivalLineupItem.jsonl       (named festival appearances)
    ra-scraper-master/scraper/EventItem.jsonl                (RA bookings)
    ra-scraper-master/scraper/lofi_booked_labels.csv         (LOFI history)
    v2-scraper/scraper/BeatportLabelArtistItem.jsonl         (label affiliations)
    v2-scraper/scraper/MixcloudEpisodeItem.jsonl             (media appearances)
    v2-scraper/scraper/RAGenreArtistItem.jsonl               (RA genre events)
"""

from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE           = Path(__file__).parent
OUT_FILE       = HERE / "scraper_data" / "artist_enriched.jsonl"

RA_SCRAPER     = HERE.parent / "ra-scraper-master" / "scraper"
V2_SCRAPER     = HERE.parent / "v2-scraper" / "scraper"
SPOTIFY_FILE   = HERE / "scraper_data" / "spotify_artists.jsonl"
SOUNDCLOUD_FILE= HERE / "scraper_data" / "soundcloud_artists.jsonl"

_TIER_ORDER = {"A+": 0, "A": 1, "B": 2}

# Known festival keywords for classification
_FESTIVAL_KEYWORDS = {
    "festival", "festiv", " festival", "outdoor", "open air", "openair",
    "awakenings", "tomorrowland", "dekmantel", "amsterdam dance event", "ade",
    "sonus", "hideout", "electric zoo", "movement", "melt", "junction 2",
    "fabric", "dc10", "circoloco", "music on", "ants", "paradise",
    "loveland", "shelter", "sexyland", "doornroosje",
}


def _slug(name: str) -> str:
    # Normalize unicode so ä→a, ö→o, ü→u etc. (prevents duplicate records per artist)
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def _legacy_slug(name: str) -> str:
    """Pre-normalization slug — used to generate old_artist_ids for backward compat."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _is_festival(event_name: str | None, venue: str | None) -> bool:
    text = ((event_name or "") + " " + (venue or "")).lower()
    return any(kw in text for kw in _FESTIVAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_lastfm_snapshots() -> dict[str, list[dict]]:
    """Returns {artist_name: [snapshots sorted by date]}."""
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(RA_SCRAPER / "lastfm" / "LastFMSnapshot.jsonl"):
        name = row.get("name") or row.get("query_name")
        if name:
            by_artist[name].append(row)
    # Sort each artist's snapshots by date
    for name in by_artist:
        by_artist[name].sort(key=lambda x: x.get("scraped_at", ""))
    return dict(by_artist)


def _unescape(v: str | None) -> str | None:
    return html.unescape(v) if v else v


def load_pf_events() -> dict[str, list[dict]]:
    """Returns {artist_name: [event dicts]} from PartyflockEventItem.jsonl."""
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(RA_SCRAPER / "PartyflockEventItem.jsonl"):
        name = row.get("artist")
        if name:
            by_artist[name].append({
                "date":       (row.get("start_date") or "")[:10],
                "event_name": _unescape(row.get("event_name")),
                "venue":      _unescape(row.get("venue")),
                "city":       _unescape(row.get("city")),
                "country":    row.get("country"),
                "url":        row.get("event_url"),
                "lat":        row.get("latitude"),
                "lon":        row.get("longitude"),
            })
    for name in by_artist:
        by_artist[name].sort(key=lambda x: x["date"])
    return dict(by_artist)


def load_pf_lineups() -> dict[str, list[dict]]:
    """Returns {artist_name_lower: [event dicts]} from PartyflockLineupItem.jsonl."""
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(RA_SCRAPER / "PartyflockLineupItem.jsonl"):
        for name in (row.get("lineup") or []):
            if name:
                by_artist[name.lower()].append({
                    "date":       (row.get("start_date") or "")[:10],
                    "event_name": _unescape(row.get("event_name")),
                    "venue":      _unescape(row.get("venue")),
                    "city":       _unescape(row.get("city")),
                    "country":    row.get("country"),
                    "url":        row.get("event_url"),
                })
    return dict(by_artist)


def load_festival_lineups() -> dict[str, list[dict]]:
    """Returns {artist_name_lower: [{festival_name, year}]}."""
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(RA_SCRAPER / "FestivalLineupItem.jsonl"):
        name = row.get("artist")
        if name:
            by_artist[name.lower()].append({
                "festival_name": row.get("festival_name"),
                "year": row.get("festival_year"),
            })
    return dict(by_artist)


def load_ra_events() -> dict[str, list[dict]]:
    """Returns {artist_name_lower: [{date, title, venue, city}]} from EventItem.jsonl."""
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(RA_SCRAPER / "EventItem.jsonl"):
        name = row.get("artist")
        if name:
            by_artist[name.lower()].append({
                "date":  row.get("date"),
                "title": row.get("title"),
                "venue": row.get("venue"),
                "city":  row.get("city"),
                "url":   row.get("link"),
            })
    return dict(by_artist)


def load_lofi_booked() -> dict[str, int]:
    """Returns {artist_name: lofi_appearance_count}."""
    booked: dict[str, int] = {}
    path = RA_SCRAPER / "lofi_booked_labels.csv"
    if not path.exists():
        return booked
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            booked[row["artist"].strip()] = int(row.get("lofi_appearance_count") or 1)
    return booked


def load_soundcloud() -> dict[str, list[dict]]:
    """Returns {artist_name_lower: [snapshots sorted by date]} from soundcloud_enricher output."""
    by_artist: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(SOUNDCLOUD_FILE):
        key = (row.get("name") or "").lower()
        if key:
            by_artist[key].append(row)
    for key in by_artist:
        by_artist[key].sort(key=lambda x: x.get("scraped_at", ""))
    return dict(by_artist)


def load_spotify_enriched() -> dict[str, dict]:
    """Returns {artist_name_lower: spotify_record} from spotify_enricher output."""
    by_artist: dict[str, dict] = {}
    for row in _read_jsonl(SPOTIFY_FILE):
        key = (row.get("name") or "").lower()
        if key:
            by_artist[key] = row
    return by_artist


def load_beatport_charts() -> dict[str, dict]:
    """Returns {artist_name_lower: {best_rank, first_top10_date, first_top1_date}}."""
    by_artist: dict[str, dict] = defaultdict(
        lambda: {"best_rank": None, "first_top10_date": None, "first_top1_date": None}
    )
    for row in _read_jsonl(V2_SCRAPER / "BeatportChartItem.jsonl"):
        rank = row.get("rank")
        if rank is None:
            continue
        date = (row.get("publish_date") or "")[:10]
        artists_raw = row.get("artists") or []
        if isinstance(artists_raw, str):
            artists_raw = [a.strip() for a in artists_raw.split(",")]
        for artist_str in artists_raw:
            key = (artist_str or "").strip().lower()
            if not key:
                continue
            cur = by_artist[key]
            if cur["best_rank"] is None or rank < cur["best_rank"]:
                cur["best_rank"] = rank
            if rank <= 10 and (cur["first_top10_date"] is None or date < cur["first_top10_date"]):
                cur["first_top10_date"] = date
            if rank == 1 and (cur["first_top1_date"] is None or date < cur["first_top1_date"]):
                cur["first_top1_date"] = date
    return dict(by_artist)


def load_beatport() -> dict[str, dict]:
    """Returns {artist_name_lower: {releases, labels, tier}}."""
    # Load label tier map
    label_tier: dict[str, str] = {}
    labels_json = V2_SCRAPER.parent / "input" / "framework_labels.json"
    if labels_json.exists():
        for l in json.loads(labels_json.read_text(encoding="utf-8")):
            if l.get("name"):
                label_tier[l["name"].lower()] = l.get("tier", "B")

    by_artist: dict[str, dict] = defaultdict(lambda: {"releases": 0, "labels": [], "tier": None})
    for row in _read_jsonl(V2_SCRAPER / "BeatportLabelArtistItem.jsonl"):
        key = row["artist_name"].lower()
        by_artist[key]["releases"] += row.get("release_count", 0)
        label = row.get("label_name")
        if label and label not in by_artist[key]["labels"]:
            by_artist[key]["labels"].append(label)
        tier = row.get("tier") or label_tier.get((label or "").lower())
        if tier:
            cur = by_artist[key]["tier"]
            if cur is None or _TIER_ORDER.get(tier, 99) < _TIER_ORDER.get(cur, 99):
                by_artist[key]["tier"] = tier
        # Store latest release date
        latest = row.get("latest_release")
        if latest:
            cur_latest = by_artist[key].get("latest_release")
            if not cur_latest or latest > cur_latest:
                by_artist[key]["latest_release"] = latest
    return dict(by_artist)


def load_mixcloud() -> dict[str, list[str]]:
    """Returns {artist_name_lower: [show_name, ...]}."""
    by_artist: dict[str, list[str]] = defaultdict(list)
    for row in _read_jsonl(V2_SCRAPER / "MixcloudEpisodeItem.jsonl"):
        show = row.get("show_name", "")
        for name in (row.get("featured_artists") or []):
            by_artist[name.lower()].append(show)
    return dict(by_artist)


def load_ra_genre() -> dict[str, dict]:
    """Returns {artist_name_lower: {event_count, genres, cities}}."""
    by_artist: dict[str, dict] = defaultdict(lambda: {"event_count": 0, "genres": [], "cities": []})
    for row in _read_jsonl(V2_SCRAPER / "RAGenreArtistItem.jsonl"):
        key = row["artist_name"].lower()
        by_artist[key]["event_count"] += row.get("event_count", 1)
        genre = row.get("genre_tag")
        if genre and genre not in by_artist[key]["genres"]:
            by_artist[key]["genres"].append(genre)
        for city in (row.get("cities") or []):
            if city and city not in by_artist[key]["cities"]:
                by_artist[key]["cities"].append(city)
    return dict(by_artist)


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def compute_growth_history(snapshots: list[dict]) -> dict:
    """Extract growth trend from multiple Last.fm snapshots."""
    if not snapshots:
        return {}
    if len(snapshots) == 1:
        s = snapshots[0]
        return {
            "current_listeners": s.get("listeners"),
            "current_playcount": s.get("playcount"),
            "snapshots": [{"date": (s.get("scraped_at") or "")[:10],
                           "listeners": s.get("listeners"),
                           "playcount": s.get("playcount")}],
            "listener_delta_total": None,
            "listener_growth_pct_total": None,
            "days_tracked": 0,
        }

    first, last = snapshots[0], snapshots[-1]
    l_first = first.get("listeners") or 0
    l_last  = last.get("listeners") or 0
    delta = l_last - l_first
    pct   = round((delta / l_first * 100), 1) if l_first else None

    # Days between first and last snapshot
    try:
        d0 = datetime.fromisoformat(first["scraped_at"].replace("Z", "+00:00"))
        d1 = datetime.fromisoformat(last["scraped_at"].replace("Z", "+00:00"))
        days = (d1 - d0).days
    except Exception:
        days = 0

    return {
        "current_listeners":         l_last,
        "current_playcount":         last.get("playcount"),
        "snapshots": [
            {"date": (s.get("scraped_at") or "")[:10],
             "listeners": s.get("listeners"),
             "playcount": s.get("playcount")}
            for s in snapshots
        ],
        "listener_delta_total":      delta,
        "listener_growth_pct_total": pct,
        "days_tracked":              days,
    }


def compute_booking_stats(events: list[dict]) -> dict:
    """Derive booking stats from PF event list."""
    if not events:
        return {"total": 0, "recent_12m": 0, "countries": [], "cities": [], "nl_ratio": None,
                "festival_count": 0, "recent_events": [], "geo_spread": 0}

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff_1y = datetime.now(timezone.utc).replace(year=datetime.now().year - 1).strftime("%Y-%m-%d")
    cutoff_2y = datetime.now(timezone.utc).replace(year=datetime.now().year - 2).strftime("%Y-%m-%d")

    countries: Counter = Counter()
    cities: Counter = Counter()
    recent_12m = 0
    prev_12m   = 0
    festival_count = 0

    for ev in events:
        date    = ev.get("date", "")
        country = ev.get("country")
        city    = ev.get("city")
        if country:
            countries[country] += 1
        if city:
            cities[city] += 1
        if date >= cutoff_1y:
            recent_12m += 1
        elif date >= cutoff_2y:
            prev_12m += 1
        if _is_festival(ev.get("event_name"), ev.get("venue")):
            festival_count += 1

    nl = countries.get("NL", 0)
    total = sum(countries.values())
    nl_ratio = round(nl / total, 2) if total else None

    # Booking velocity: recent vs previous year
    velocity = None
    if prev_12m > 0:
        velocity = round(recent_12m / prev_12m, 2)
    elif recent_12m > 0:
        velocity = 2.0   # all growth, no baseline

    return {
        "total":          total,
        "recent_12m":     recent_12m,
        "prev_12m":       prev_12m,
        "booking_velocity": velocity,   # >1 = growing, <1 = declining
        "countries":      [c for c, _ in countries.most_common(10)],
        "cities":         [c for c, _ in cities.most_common(10)],
        "nl_events":      nl,
        "nl_ratio":       nl_ratio,
        "geo_spread":     len(countries),
        "festival_count": festival_count,
        "recent_events": sorted(events, key=lambda x: x["date"], reverse=True)[:8],
    }


def compute_momentum_score(
    growth_history: dict,
    booking_stats: dict,
    beatport: dict | None,
    mixcloud_count: int,
    ra_event_count: int,
) -> float:
    """
    Composite 0-100 momentum score.
    Weights: booking velocity 30%, listener growth 25%, geo spread 20%,
             label activity 15%, media presence 10%.
    """
    score = 0.0

    # Booking velocity (30%): >1 growing, cap at 3x
    velocity = booking_stats.get("booking_velocity") or 0
    score += min(velocity / 3.0, 1.0) * 30

    # Listener growth rate (25%): normalize to +/-50% range
    pct = growth_history.get("listener_growth_pct_total") or 0
    norm_growth = min(max((pct + 50) / 100, 0), 1)
    score += norm_growth * 25

    # Geographic spread (20%): 5+ countries = full score
    geo = booking_stats.get("geo_spread", 0)
    score += min(geo / 5, 1.0) * 20

    # Beatport label activity (15%): A+ tier with recent release
    if beatport:
        tier_pts = {"A+": 1.0, "A": 0.7, "B": 0.4}.get(beatport.get("tier") or "", 0)
        releases_pts = min((beatport.get("releases") or 0) / 20, 1.0)
        score += (tier_pts * 0.6 + releases_pts * 0.4) * 15

    # Media presence (10%): Mixcloud + RA genre
    media = min((mixcloud_count + ra_event_count) / 10, 1.0)
    score += media * 10

    return round(score, 1)


# ---------------------------------------------------------------------------
# Milestone detection
# ---------------------------------------------------------------------------

_MILESTONE_KEYS = [
    "first_ibiza", "first_circoloco", "first_music_on", "first_ants",
    "first_piv_release", "first_beatport_top10", "first_beatport_no1",
    "first_festival",
    # Manual-only (auto-set to None, filled via feedback UI later)
    "first_boiler_room", "first_ra_podcast", "first_bbc_r1",
    "first_headline_500", "first_headline_1000", "first_headline_2000", "first_headline_5000",
    "first_tier_a_support", "first_tier_a_b2b",
    "first_extended_set", "first_anl", "first_adl",
    "first_major_residency", "first_multi_city_tour",
]


def compute_milestones(
    pf_events: list[dict],
    ra_events: list[dict],
    beatport: dict | None,
    beatport_charts: dict | None,
    festival_history: list[dict],
) -> dict:
    milestones: dict = {k: None for k in _MILESTONE_KEYS}

    # Merge and sort all events by date; decode HTML entities from scraper output
    all_events = []
    for ev in pf_events:
        all_events.append({
            "date":  ev.get("date", ""),
            "city":  html.unescape(ev.get("city") or "").lower(),
            "text":  html.unescape((ev.get("event_name") or "") + " " + (ev.get("venue") or "")).lower(),
        })
    for ev in ra_events:
        all_events.append({
            "date":  (ev.get("date") or ""),
            "city":  html.unescape(ev.get("city") or "").lower(),
            "text":  html.unescape((ev.get("title") or "") + " " + (ev.get("venue") or "")).lower(),
        })
    all_events.sort(key=lambda x: x["date"])

    for ev in all_events:
        date = ev["date"]
        city = ev["city"]
        text = ev["text"]
        if "ibiza" in city and milestones["first_ibiza"] is None:
            milestones["first_ibiza"] = date
        if ("circoloco" in text or "dc10" in text) and milestones["first_circoloco"] is None:
            milestones["first_circoloco"] = date
        if "music on" in text and milestones["first_music_on"] is None:
            milestones["first_music_on"] = date
        if re.search(r"\bants\b", text) and milestones["first_ants"] is None:
            milestones["first_ants"] = date
        # B2B detection — event name contains "b2b" pattern
        if " b2b " in text and milestones["first_tier_a_b2b"] is None:
            milestones["first_tier_a_b2b"] = date

    # PIV label
    for label in (beatport or {}).get("labels") or []:
        if "piv" in label.lower():
            milestones["first_piv_release"] = (beatport or {}).get("latest_release")
            break

    # Beatport charts
    if beatport_charts:
        milestones["first_beatport_top10"] = beatport_charts.get("first_top10_date")
        milestones["first_beatport_no1"]   = beatport_charts.get("first_top1_date")

    # First named festival
    if festival_history:
        milestones["first_festival"] = festival_history[0].get("festival_name")

    return milestones


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def build_enriched_artist(
    name: str,
    lastfm_snapshots: list[dict],
    pf_events: list[dict],
    pf_lineup_events: list[dict],
    festival_history: list[dict],
    ra_events: list[dict],
    lofi_count: int,
    beatport: dict | None,
    beatport_charts: dict | None,
    mixcloud_shows: list[str],
    ra_genre: dict | None,
    spotify: dict | None,
    spotify_enriched: dict | None,
    sc_snapshots: list[dict] | None = None,
) -> dict:
    growth     = compute_growth_history(lastfm_snapshots)
    booking    = compute_booking_stats(pf_events or pf_lineup_events)
    last_snap  = lastfm_snapshots[-1] if lastfm_snapshots else {}
    ra_count   = (ra_genre or {}).get("event_count", 0)
    mc_count   = len(mixcloud_shows)
    momentum   = compute_momentum_score(growth, booking, beatport, mc_count, ra_count)

    # Deduplicated festival list: named + keyword-detected
    festivals_named = [f["festival_name"] for f in festival_history if f.get("festival_name")]
    festivals_detected = list({
        ev.get("event_name") or ev.get("venue", "")
        for ev in (pf_events or [])
        if _is_festival(ev.get("event_name"), ev.get("venue"))
    })
    all_festivals = list(dict.fromkeys(festivals_named + festivals_detected))[:20]

    milestones = compute_milestones(
        pf_events       = pf_events or [],
        ra_events       = ra_events or [],
        beatport        = beatport,
        beatport_charts = beatport_charts,
        festival_history= festival_history,
    )

    return {
        "artist_id":   _slug(name),
        "name":        name,
        "scraped_at":  datetime.now(timezone.utc).isoformat(),

        # Growth history
        "growth_history":               growth,
        "lastfm_tags":                  last_snap.get("tags") or [],
        "lastfm_similar":               last_snap.get("similar") or [],
        "lastfm_listener_growth_30d_pct": last_snap.get("listener_growth_pct"),   # from features.csv col
        "pf_fans":                      last_snap.get("pf_fans"),

        # Momentum
        "momentum_score":               momentum,

        # Booking history
        "booking_stats":                booking,
        "ra_events":                    ra_events[:20],

        # Festival history
        "festival_history":             all_festivals,

        # Label affiliations
        "beatport_releases":            (beatport or {}).get("releases"),
        "beatport_labels":              (beatport or {}).get("labels") or [],
        "beatport_label_tier":          (beatport or {}).get("tier"),
        "beatport_latest_release":      (beatport or {}).get("latest_release"),

        # Agency affiliations (not yet scraped)
        "agency":                       None,
        "agency_tier":                  None,

        # Geographic
        "geo_countries":                booking.get("countries") or [],
        "geo_cities":                   booking.get("cities") or [],
        "nl_ratio":                     booking.get("nl_ratio"),
        "geo_spread":                   booking.get("geo_spread", 0),

        # Media
        "mixcloud_shows":               list(dict.fromkeys(mixcloud_shows))[:10],
        "mixcloud_appearances":         mc_count,
        "ra_genre_events":              ra_count,
        "ra_genres":                    (ra_genre or {}).get("genres") or [],
        "ra_cities":                    (ra_genre or {}).get("cities") or [],

        # Spotify (prefer spotify_enricher data, fall back to old scraper)
        "spotify_id":          (spotify_enriched or spotify or {}).get("spotify_id"),
        "spotify_url":         (spotify_enriched or spotify or {}).get("spotify_url"),
        "spotify_followers":   (spotify_enriched or {}).get("followers") or (spotify or {}).get("followers"),
        "spotify_popularity":  (spotify_enriched or {}).get("popularity"),
        "spotify_genres":      (spotify_enriched or {}).get("genres") or (spotify or {}).get("genres") or [],
        "spotify_related":     (spotify_enriched or {}).get("related_artists") or [],
        "image_url":           (spotify_enriched or {}).get("image_url") or (spotify_enriched or {}).get("image_url_thumb"),
        "image_url_thumb":     (spotify_enriched or {}).get("image_url_thumb"),

        # SoundCloud (latest snapshot + full history for time series)
        "sc_snapshots": [
            {"date": (s.get("scraped_at") or "")[:10],
             "followers": s.get("followers"),
             "tracks": s.get("tracks")}
            for s in (sc_snapshots or [])
        ],
        "sc_followers":  (sc_snapshots[-1].get("followers") if sc_snapshots else None),
        "sc_tracks":     (sc_snapshots[-1].get("tracks") if sc_snapshots else None),
        "sc_url":        (sc_snapshots[-1].get("sc_url") if sc_snapshots else None),
        "sc_username":   (sc_snapshots[-1].get("sc_username") if sc_snapshots else None),

        # LOFI
        "lofi_booked":                  lofi_count > 0,
        "lofi_appearance_count":        lofi_count,
        "lofi_feedback_history":        [],   # populated by Tinder swipes at runtime

        # Milestones
        "milestones":                   milestones,
    }


def run_aggregation(verbose: bool = True) -> int:
    if verbose:
        print("Loading source data...")

    lastfm_snaps      = load_lastfm_snapshots()
    pf_events         = load_pf_events()
    pf_lineups        = load_pf_lineups()
    festival_lups     = load_festival_lineups()
    ra_events_map     = load_ra_events()
    lofi_booked       = load_lofi_booked()
    beatport_map      = load_beatport()
    beatport_charts   = load_beatport_charts()
    mixcloud_map      = load_mixcloud()
    ra_genre_map      = load_ra_genre()
    spotify_enriched  = load_spotify_enriched()
    soundcloud_map    = load_soundcloud()

    # Load Spotify if available
    spotify_map: dict[str, dict] = {}
    spotify_path = RA_SCRAPER / "spotify" / "SpotifyArtistItem.jsonl"
    for row in _read_jsonl(spotify_path):
        key = (row.get("query_name") or row.get("name", "")).lower()
        if key:
            spotify_map[key] = row

    # Collect all raw names across all sources, then group by slug to merge duplicates.
    # This handles artists spelled differently across sources (e.g. umlaut variants).
    raw_names: set[str] = set()
    raw_names.update(lastfm_snaps.keys())
    raw_names.update(pf_events.keys())
    raw_names.update(lofi_booked.keys())
    for key in list(beatport_map.keys()) + list(mixcloud_map.keys()):
        match = next((n for n in raw_names if n.lower() == key), None)
        if not match:
            raw_names.add(key.title())

    # Group raw names by slug; canonical = longest name (preserves special chars)
    slug_to_names: dict[str, list[str]] = defaultdict(list)
    for name in raw_names:
        slug_to_names[_slug(name)].append(name)
    canonical_names: dict[str, str] = {
        slug: max(names, key=len)
        for slug, names in slug_to_names.items()
    }   # slug → display name

    if verbose:
        merged_count = sum(1 for names in slug_to_names.values() if len(names) > 1)
        print(f"  Artists to aggregate: {len(canonical_names)} ({merged_count} slug-merged)")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for slug in sorted(canonical_names):
            name    = canonical_names[slug]
            aliases = slug_to_names[slug]   # all raw names that map to this slug

            # Merge source data from all aliases
            lfm_snaps: list[dict] = []
            pf_evs:    list[dict] = []
            lofi_cnt   = 0
            for alias in aliases:
                lfm_snaps.extend(lastfm_snaps.get(alias) or [])
                pf_evs.extend(pf_events.get(alias) or [])
                lofi_cnt += lofi_booked.get(alias, 0)
            # Sort merged lastfm snapshots by date; deduplicate same-date entries
            # by keeping the highest-listener record (avoids wrong-match artifacts)
            lfm_snaps.sort(key=lambda x: x.get("scraped_at", ""))
            seen_dates: dict[str, dict] = {}
            for s in lfm_snaps:
                d = (s.get("scraped_at") or "")[:10]
                if d not in seen_dates or (s.get("listeners") or 0) > (seen_dates[d].get("listeners") or 0):
                    seen_dates[d] = s
            lfm_snaps = sorted(seen_dates.values(), key=lambda x: x.get("scraped_at", ""))

            key = name.lower()
            record = build_enriched_artist(
                name              = name,
                lastfm_snapshots  = lfm_snaps,
                pf_events         = pf_evs,
                pf_lineup_events  = pf_lineups.get(key) or [],
                festival_history  = festival_lups.get(key) or [],
                ra_events         = ra_events_map.get(key) or [],
                lofi_count        = lofi_cnt,
                beatport          = beatport_map.get(key),
                beatport_charts   = beatport_charts.get(key),
                mixcloud_shows    = mixcloud_map.get(key) or [],
                ra_genre          = ra_genre_map.get(key),
                spotify           = spotify_map.get(key),
                spotify_enriched  = spotify_enriched.get(key),
                sc_snapshots      = soundcloud_map.get(key) or [],
            )
            # Store legacy slugs (pre-unicode-normalization) so the app can look up
            # artists by any artist_id that existed before the slug fix
            old_ids = sorted({_legacy_slug(a) for a in aliases} - {slug})
            if old_ids:
                record["old_artist_ids"] = old_ids
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if verbose:
        print(f"Written: {count} enriched artist records -> {OUT_FILE}")
    return count


if __name__ == "__main__":
    run_aggregation()
