"""
Sentence-transformer embeddings + FAISS index.

All vectors are L2-normalised so inner product == cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import ArtistProfile

_DATA_DIR = Path(__file__).parent.parent / "data"
_INDEX_FILE = _DATA_DIR / "faiss.index"
_ID_MAP_FILE = _DATA_DIR / "faiss_id_map.json"   # index position -> artist_id
_CENTROID_FILE = _DATA_DIR / "lofi_centroid.npy"
_FEATURE_CENTROID_FILE = _DATA_DIR / "lofi_feature_centroid.npy"
_FEATURE_CENTROID_CORE_FILE     = _DATA_DIR / "lofi_feature_centroid_core.npy"
_FEATURE_CENTROID_EMERGING_FILE = _DATA_DIR / "lofi_feature_centroid_emerging.npy"

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DIM = 384
_FEATURE_DIM = 14

# Feature dimension names — each maps to a Chartmetric API filter param
_FEATURE_NAMES = [
    "log_listeners",      # → min/max_spotify_monthly_listeners
    "listener_growth",    # → min_listener_growth_90d_pct
    "momentum_score",     # → composite quality signal
    "booking_total",      # → career stage proxy
    "booking_velocity",   # → trend direction (>1 = growing)
    "recent_bookings",    # → min_bookings_12m
    "geo_spread",         # → min_countries
    "nl_ratio",           # → geographic focus (NL/EU weighted)
    "beatport_releases",  # → release activity
    "bp_tier",            # → label tier (A+=1.0, A=0.7, B=0.4)
    "mixcloud",           # → podcast/media presence
    "ra_events",          # → underground credibility
    "festival_count",     # → festival history depth
    "pf_fans",            # → local NL demand
    # NOTE: lofi_booked intentionally excluded — it IS the label, not a predictive feature.
    # Including it in the centroid and then ranking candidates (all lofi_booked=0) against
    # it creates tautological leakage where every candidate is penalised on one dimension
    # by design. The signal is captured by booking_total and booking_velocity instead.
]

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype("float32")


def embed_profiles(profiles: list[ArtistProfile]) -> list[ArtistProfile]:
    texts = [p.profile_text for p in profiles]
    vecs = embed_texts(texts)
    for profile, vec in zip(profiles, vecs):
        profile.embedding = vec.tolist()
    return profiles


def build_index(profiles: list[ArtistProfile]) -> tuple[faiss.IndexFlatIP, list[str]]:
    """Build a new FAISS index from profiles. Returns (index, artist_id_list)."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    vecs = np.array([p.embedding for p in profiles], dtype="float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(_DIM)
    index.add(vecs)

    id_map = [p.artist_id for p in profiles]
    return index, id_map


def save_index(index: faiss.IndexFlatIP, id_map: list[str]) -> None:
    import json
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_FILE))
    _ID_MAP_FILE.write_text(json.dumps(id_map), encoding="utf-8")


def load_index() -> tuple[faiss.IndexFlatIP, list[str]]:
    import json
    if not _INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found at {_INDEX_FILE}. Run: python run.py --seed")
    index = faiss.read_index(str(_INDEX_FILE))
    id_map = json.loads(_ID_MAP_FILE.read_text(encoding="utf-8"))
    return index, id_map


def compute_centroid(profile_vecs: list[list[float]]) -> np.ndarray:
    vecs = np.array(profile_vecs, dtype="float32")
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm
    return centroid


def save_centroid(centroid: np.ndarray) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(_CENTROID_FILE), centroid)


def load_centroid() -> np.ndarray | None:
    if not _CENTROID_FILE.exists():
        return None
    return np.load(str(_CENTROID_FILE))


def cosine_dist_to_centroid(embedding: list[float], centroid: np.ndarray) -> float:
    vec = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    similarity = float(np.dot(vec, centroid))
    return 1.0 - similarity   # distance: 0 = identical, 2 = opposite


# ---------------------------------------------------------------------------
# Structured feature centroid (15-dim numeric, directly maps to API params)
# ---------------------------------------------------------------------------

def extract_feature_vector(enriched: dict) -> np.ndarray:
    """15-dim normalised [0,1] feature vector derived from an enriched artist record."""
    import math

    gh = enriched.get("growth_history") or {}
    bs = enriched.get("booking_stats") or {}

    listeners = gh.get("current_listeners") or enriched.get("spotify_followers") or 0
    log_listeners = math.log10(max(listeners, 1)) / 7.0  # log10(10M) = 7

    growth = gh.get("listener_growth_pct_total") or 0.0
    listener_growth_norm = min(max((growth + 100.0) / 200.0, 0.0), 1.0)

    momentum = (enriched.get("momentum_score") or 0.0) / 100.0

    total_bookings = bs.get("total") or 0
    booking_total_norm = math.log10(max(total_bookings, 1)) / math.log10(501)

    velocity = bs.get("booking_velocity")
    booking_velocity_norm = min((velocity or 1.0) / 3.0, 1.0)

    recent = bs.get("recent_12m") or 0
    recent_bookings_norm = math.log10(max(recent, 1)) / math.log10(101)

    geo_spread = (bs.get("geo_spread") or enriched.get("geo_spread") or 0) / 50.0

    nl_ratio = float(bs.get("nl_ratio") or enriched.get("nl_ratio") or 0.0)

    bp_releases = enriched.get("beatport_releases") or 0
    bp_releases_norm = math.log10(max(bp_releases, 1)) / math.log10(201)

    bp_tier = (enriched.get("beatport_label_tier") or "").upper()
    bp_tier_enc = {"A+": 1.0, "A": 0.7, "B": 0.4}.get(bp_tier, 0.0)

    mc = enriched.get("mixcloud_appearances") or 0
    mc_norm = math.log10(max(mc, 1)) / math.log10(51)

    ra = enriched.get("ra_genre_events") or 0
    ra_norm = math.log10(max(ra, 1)) / math.log10(10001)

    fest = bs.get("festival_count") or 0
    fest_norm = math.log10(max(fest, 1)) / math.log10(51)

    pf = enriched.get("pf_fans") or 0
    pf_norm = math.log10(max(pf, 1)) / math.log10(100001)

    return np.array([
        log_listeners, listener_growth_norm, momentum,
        booking_total_norm, booking_velocity_norm, recent_bookings_norm,
        geo_spread, nl_ratio,
        bp_releases_norm, bp_tier_enc,
        mc_norm, ra_norm, fest_norm, pf_norm,
    ], dtype="float32")


def compute_feature_centroid(feature_vecs: list[np.ndarray]) -> np.ndarray:
    """Simple mean — values are already in [0,1], no L2-norm needed."""
    arr = np.stack(feature_vecs, axis=0)
    return arr.mean(axis=0).astype("float32")


def save_feature_centroid(centroid: np.ndarray) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(_FEATURE_CENTROID_FILE), centroid)


def load_feature_centroid() -> np.ndarray | None:
    if not _FEATURE_CENTROID_FILE.exists():
        return None
    return np.load(str(_FEATURE_CENTROID_FILE))


def compute_dual_feature_centroids(
    enriched_list: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split LOFI-booked artists into two groups by career stage (median total
    bookings as threshold) and return (core_centroid, emerging_centroid).

    Core     = top half by total bookings — established artists LOFI already books regularly.
    Emerging = bottom half — underground/rising artists LOFI discovered early.

    Ranking against BOTH centroids (nearest-cluster wins) lets the system surface
    both established fits and early-stage discoveries rather than averaging them into
    one muddled centroid.
    """
    vecs_and_bookings: list[tuple[int, np.ndarray]] = []
    for enr in enriched_list:
        bs    = enr.get("booking_stats") or {}
        total = int(bs.get("total") or 0)
        vec   = extract_feature_vector(enr)
        vecs_and_bookings.append((total, vec))

    if not vecs_and_bookings:
        zero = np.zeros(_FEATURE_DIM, dtype="float32")
        return zero, zero

    sorted_totals = sorted(t for t, _ in vecs_and_bookings)
    median_idx    = len(sorted_totals) // 2
    threshold     = sorted_totals[median_idx]

    core_vecs     = [v for t, v in vecs_and_bookings if t >= threshold]
    emerging_vecs = [v for t, v in vecs_and_bookings if t < threshold]

    core     = np.stack(core_vecs).mean(0).astype("float32")     if core_vecs     else np.zeros(_FEATURE_DIM, dtype="float32")
    emerging = np.stack(emerging_vecs).mean(0).astype("float32") if emerging_vecs else np.zeros(_FEATURE_DIM, dtype="float32")
    return core, emerging


def save_dual_feature_centroids(core: np.ndarray, emerging: np.ndarray) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(_FEATURE_CENTROID_CORE_FILE), core)
    np.save(str(_FEATURE_CENTROID_EMERGING_FILE), emerging)


def load_dual_feature_centroids() -> tuple[np.ndarray | None, np.ndarray | None]:
    core     = np.load(str(_FEATURE_CENTROID_CORE_FILE))     if _FEATURE_CENTROID_CORE_FILE.exists()     else None
    emerging = np.load(str(_FEATURE_CENTROID_EMERGING_FILE)) if _FEATURE_CENTROID_EMERGING_FILE.exists() else None
    return core, emerging


def chartmetric_params_from_feature_centroid(
    centroid: np.ndarray,
    std: np.ndarray | None = None,
) -> dict:
    """
    Translate the 14-dim structured centroid into Chartmetric API filter params.
    'std' controls the search window around each dimension (default ±0.15).
    """
    if std is None:
        std = np.full(_FEATURE_DIM, 0.15, dtype="float32")

    # Decode log-normalised listener range back to real numbers
    lo_norm = max(0.0, float(centroid[0]) - float(std[0]))
    hi_norm = min(1.0, float(centroid[0]) + float(std[0]))
    listeners_min = int(10 ** (lo_norm * 7))
    listeners_max = int(10 ** (hi_norm * 7))

    # Decode growth back to percentage
    growth_lo = (float(centroid[1]) - float(std[1])) * 200.0 - 100.0
    growth_hi = (float(centroid[1]) + float(std[1])) * 200.0 - 100.0

    label_tier_min = None
    if centroid[9] >= 0.6:
        label_tier_min = "A"
    elif centroid[9] >= 0.3:
        label_tier_min = "B"

    return {
        "min_spotify_monthly_listeners": listeners_min,
        "max_spotify_monthly_listeners": listeners_max,
        "min_listener_growth_90d_pct": round(growth_lo, 1),
        "max_listener_growth_90d_pct": round(growth_hi, 1),
        "min_bookings_12m": max(1, int(centroid[5] ** 2 * 100)),
        "min_geo_spread_countries": max(1, int(float(centroid[6]) * 50)),
        "label_tier_min": label_tier_min,
        "_raw_centroid": {
            name: round(float(v), 3)
            for name, v in zip(_FEATURE_NAMES, centroid)
        },
    }
