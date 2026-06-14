"""
LinUCB contextual bandit for artist preference learning.

Context  = 384-dim embedding vector (same as cosine ranking)
Reward   = +1 for YES swipe, -1 for NO swipe
Output   = per-artist score that blends into the cosine ranking

LinUCB paper: Li et al. 2010 "A Contextual-Bandit Approach to Personalized News Article Recommendation"
We use a single shared arm (disjoint model) since each artist is only seen once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent.parent / "data"
_WEIGHTS_FILE = _DATA_DIR / "mab_weights.npz"

_DIM = 14      # 14-dim structured feature vector (lofi_booked excluded — it's the label, not a feature)
_ALPHA = 0.1   # low exploration — trust the structured-feature prior early on


class LinUCB:
    def __init__(self, alpha: float = _ALPHA, dim: int = _DIM) -> None:
        self.alpha = alpha
        self.dim = dim
        self.A = np.eye(dim, dtype="float64")       # dim x dim
        self.b = np.zeros((dim, 1), dtype="float64") # dim x 1

    def update(self, context: np.ndarray, reward: float) -> None:
        x = context.reshape(-1, 1).astype("float64")
        self.A += x @ x.T
        self.b += reward * x

    def score(self, context: np.ndarray) -> float:
        x = context.reshape(-1, 1).astype("float64")
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        ucb = float((theta.T @ x).item()) + self.alpha * float(np.sqrt((x.T @ A_inv @ x).item()))
        return ucb

    def score_batch(self, embeddings: dict[str, list[float]]) -> dict[str, float]:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        scores = {}
        for artist_id, emb in embeddings.items():
            x = np.array(emb, dtype="float64").reshape(-1, 1)
            ucb = float((theta.T @ x).item()) + self.alpha * float(np.sqrt((x.T @ A_inv @ x).item()))
            scores[artist_id] = float(np.clip(ucb, -2.0, 2.0))
        return scores

    def save(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(str(_WEIGHTS_FILE), A=self.A, b=self.b)

    @classmethod
    def load(cls, alpha: float = _ALPHA) -> "LinUCB":
        mab = cls(alpha=alpha)
        if _WEIGHTS_FILE.exists():
            try:
                data = np.load(str(_WEIGHTS_FILE))
                if data["A"].shape == (_DIM, _DIM):
                    mab.A = data["A"]
                    mab.b = data["b"]
                else:
                    # Dimension mismatch (e.g. old 384-dim weights) — start fresh
                    print(
                        f"[MAB] Weight shape {data['A'].shape} != ({_DIM},{_DIM}). "
                        "Starting fresh. Run: python run.py --retrain-mab"
                    )
            except Exception:
                pass
        return mab


def reward_for_decision(decision: str) -> float | None:
    """Graded rewards so the MAB learns nuance across all label types."""
    return {
        "yes":          1.0,
        "monitor":      0.25,   # soft positive — interesting but uncertain
        "not_ready":   -0.1,   # temporal, almost neutral
        "saturated_nl": -0.5,  # context-specific, not an inherent flaw
        "no":          -0.5,   # generic rejection
        "wrong_genre": -0.75,  # style mismatch
        "commercial":  -1.0,   # strong rejection — opposite of LOFI taste
        "skip":         None,  # no update
    }.get(decision)
