"""
Ranks candidate artists by cosine distance to the LOFI taste centroid,
blended with MAB LinUCB score once enough swipes have accumulated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embedder import (
    cosine_dist_to_centroid, extract_feature_vector,
    load_centroid, load_feature_centroid, load_dual_feature_centroids,
)
from .schemas import ArtistProfile, SwipeRecord

_DATA_DIR = Path(__file__).parent.parent / "data"
_CANDIDATES_FILE = _DATA_DIR / "candidates.jsonl"
_SWIPES_FILE = _DATA_DIR / "swipes.jsonl"

# Primary ranking: structured feature similarity (15-dim, interpretable)
# MAB: also trained on feature vectors — blended in once ≥20 swipes exist
_FEATURE_WEIGHT = 0.7
_MAB_WEIGHT = 0.3


def _cosine_sim(vec: np.ndarray, centroid: np.ndarray) -> float:
    vn = np.linalg.norm(vec)
    cn = np.linalg.norm(centroid)
    if vn == 0 or cn == 0:
        return 0.0
    return float(np.dot(vec, centroid) / (vn * cn))


def load_candidates() -> list[dict]:
    if not _CANDIDATES_FILE.exists():
        return []
    rows = []
    for line in _CANDIDATES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_swipes() -> list[SwipeRecord]:
    if not _SWIPES_FILE.exists():
        return []
    swipes = []
    for line in _SWIPES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                swipes.append(SwipeRecord(**json.loads(line)))
            except Exception:
                pass
    return swipes


def get_swiped_ids(swipes: list[SwipeRecord]) -> set[str]:
    return {s.artist_id for s in swipes if s.decision != "skip"}


def rank_candidates(
    profiles: dict[str, ArtistProfile],
    swiped_ids: set[str],
    enriched_map: dict[str, dict],
    mab_scores: dict[str, float] | None = None,
    limit: int = 20,
) -> list[ArtistProfile]:
    core_centroid, emerging_centroid = load_dual_feature_centroids()
    single_centroid = load_feature_centroid()   # combined mean, fallback
    text_centroid   = load_centroid()            # 384-dim fallback for discovered artists

    ranked = []
    for artist_id, profile in profiles.items():
        if artist_id in swiped_ids:
            continue
        if not profile.embedding:
            continue

        enriched = enriched_map.get(artist_id) or {}

        if enriched:
            fvec = extract_feature_vector(enriched)

            if core_centroid is not None and emerging_centroid is not None:
                # Dual-centroid: score against both clusters, nearest cluster wins.
                # This lets established and emerging artists both surface without
                # one averaging out the other.
                core_sim     = _cosine_sim(fvec, core_centroid)
                emerging_sim = _cosine_sim(fvec, emerging_centroid)
                feat_sim     = max(core_sim, emerging_sim)
                # Store which cluster this artist is closest to for display
                profile.nearest_cluster = "core" if core_sim >= emerging_sim else "emerging"
            elif single_centroid is not None:
                feat_sim = _cosine_sim(fvec, single_centroid)
            else:
                feat_sim = 0.5

        elif text_centroid is not None:
            # Discovered artists with no enriched data: fall back to text embedding
            dist     = cosine_dist_to_centroid(profile.embedding, text_centroid)
            feat_sim = 1.0 - dist
        else:
            feat_sim = 0.5

        profile.cosine_dist_to_centroid = 1.0 - feat_sim

        mab = mab_scores.get(artist_id, 0.0) if mab_scores else 0.0
        final_score = _FEATURE_WEIGHT * feat_sim + _MAB_WEIGHT * mab

        ranked.append((final_score, profile))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:limit]]
