"""
Pydantic schemas for the LOFI Tinder discovery system.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GrowthHistory(BaseModel):
    current_listeners: int | None = None
    current_playcount: int | None = None
    snapshots: list[dict] = Field(default_factory=list)   # [{date, listeners, playcount}]
    listener_delta_total: int | None = None
    listener_growth_pct_total: float | None = None
    days_tracked: int = 0


class BookingStats(BaseModel):
    total: int = 0
    recent_12m: int = 0
    prev_12m: int = 0
    booking_velocity: float | None = None   # >1 growing, <1 declining
    countries: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    nl_events: int = 0
    nl_ratio: float | None = None
    geo_spread: int = 0
    festival_count: int = 0
    recent_events: list[dict] = Field(default_factory=list)


class ArtistEnriched(BaseModel):
    """Full enriched artist record — output of data_aggregator.py."""
    artist_id: str
    name: str
    scraped_at: str = ""

    # Growth history (Last.fm time series)
    growth_history: dict = Field(default_factory=dict)
    lastfm_tags: list[str] = Field(default_factory=list)
    lastfm_similar: list[str] = Field(default_factory=list)
    lastfm_listener_growth_30d_pct: float | None = None
    pf_fans: int | None = None

    # Momentum
    momentum_score: float = 0.0

    # Booking history
    booking_stats: dict = Field(default_factory=dict)
    ra_events: list[dict] = Field(default_factory=list)

    # Festival history
    festival_history: list[str] = Field(default_factory=list)

    # Label affiliations
    beatport_releases: int | None = None
    beatport_labels: list[str] = Field(default_factory=list)
    beatport_label_tier: str | None = None
    beatport_latest_release: str | None = None

    # Agency affiliations (not yet scraped — future)
    agency: str | None = None
    agency_tier: str | None = None

    # Geographic
    geo_countries: list[str] = Field(default_factory=list)
    geo_cities: list[str] = Field(default_factory=list)
    nl_ratio: float | None = None
    geo_spread: int = 0

    # Media
    mixcloud_shows: list[str] = Field(default_factory=list)
    mixcloud_appearances: int = 0
    ra_genre_events: int = 0
    ra_genres: list[str] = Field(default_factory=list)
    ra_cities: list[str] = Field(default_factory=list)

    # Spotify
    spotify_id: str | None = None
    spotify_url: str | None = None
    spotify_followers: int | None = None
    spotify_popularity: int | None = None
    spotify_genres: list[str] = Field(default_factory=list)

    # LOFI
    lofi_booked: bool = False
    lofi_appearance_count: int = 0
    lofi_feedback_history: list[dict] = Field(default_factory=list)


class ArtistInput(BaseModel):
    """Slimmed-down input to the LLM profile builder — derived from ArtistEnriched."""
    artist_id: str
    name: str
    enriched: dict = Field(default_factory=dict)   # full ArtistEnriched dict for app display


class ArtistProfile(BaseModel):
    artist_id: str
    name: str
    profile_text: str
    embedding: list[float]
    cosine_dist_to_centroid: float = 1.0
    nearest_cluster: str = "unknown"   # "core" | "emerging" | "unknown"
    generated_at: str


class SwipeRecord(BaseModel):
    artist_id: str
    name: str
    # "yes"/"no"/"skip" — legacy + kept for simple cases
    # Richer negative labels become structured ML features later:
    #   commercial  → Spotify popularity / listener scale too high
    #   wrong_genre → tag mismatch (detectable post-hoc from lastfm_tags)
    #   saturated_nl → nl_ratio already computed; too many NL bookings
    #   not_ready   → temporal signal; artist needs more time
    #   monitor     → soft positive; resurface in future batches
    decision: Literal["yes", "no", "skip", "commercial", "wrong_genre", "saturated_nl", "not_ready", "monitor"]
    ts: str
    cosine_dist_at_swipe: float
    linucb_score_at_swipe: float = 0.0
    profile_text: str
