"""
Neo4j client — persistent swipe storage and artist graph.

Falls back gracefully if Neo4j is unavailable (driver not installed,
URI not configured, or network error).

URI derivation: if NEO4J_URI is not set but NEO4J_USER looks like an Aura
instance ID (8 hex chars), we construct:
    neo4j+s://{NEO4J_USER}.databases.neo4j.io

Aura default username is "neo4j" — we try that first, then fall back to
the configured NEO4J_USER value.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:
    from neo4j import GraphDatabase as _GDB
    _DRIVER_AVAILABLE = True
except ImportError:
    _DRIVER_AVAILABLE = False

if TYPE_CHECKING:
    pass


def _derive_uri() -> str:
    uri = os.environ.get("NEO4J_URI", "").strip()
    if uri:
        return uri
    user = os.environ.get("NEO4J_USER", "").strip()
    if re.fullmatch(r"[0-9a-f]{8}", user, re.IGNORECASE):
        return f"neo4j+s://{user}.databases.neo4j.io"
    return ""


class Neo4jClient:
    """Thin wrapper around the Neo4j driver. All public methods are no-ops if unavailable."""

    def __init__(self) -> None:
        self._driver = None
        if not _DRIVER_AVAILABLE:
            return
        uri = _derive_uri()
        if not uri:
            return
        pwd = os.environ.get("NEO4J_PASSWORD", "").strip()
        if not pwd:
            return
        user = os.environ.get("NEO4J_USER", "neo4j").strip()
        # Aura default username is "neo4j"; configured user may be the instance ID.
        for try_user in dict.fromkeys(["neo4j", user]):
            try:
                drv = _GDB.driver(uri, auth=(try_user, pwd))
                drv.verify_connectivity()
                self._driver = drv
                break
            except Exception:
                continue
        if self._driver:
            self._ensure_schema()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._driver is not None

    def save_swipe(
        self,
        artist_id: str,
        name: str,
        decision: str,
        ts: str,
        score: float = 0.0,
        profile_text: str = "",
    ) -> None:
        if not self._driver:
            return
        swipe_id = f"{artist_id}::{ts}"
        with self._driver.session() as s:
            s.run(
                """
                MERGE (a:Artist {artist_id: $aid})
                  ON CREATE SET a.name = $name
                  ON MATCH  SET a.name = $name
                CREATE (sw:Swipe {
                    swipe_id:    $swipe_id,
                    decision:    $decision,
                    ts:          $ts,
                    score:       $score,
                    profile_text: $profile_text
                })
                CREATE (a)-[:RECEIVED_SWIPE]->(sw)
                """,
                aid=artist_id, name=name, swipe_id=swipe_id,
                decision=decision, ts=ts, score=score, profile_text=profile_text,
            )

    def load_swipes(self) -> list[dict]:
        """Return list of swipe dicts ordered by ts ascending."""
        if not self._driver:
            return []
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (a:Artist)-[:RECEIVED_SWIPE]->(sw:Swipe)
                RETURN a.artist_id AS artist_id,
                       a.name      AS name,
                       sw.decision AS decision,
                       sw.ts       AS ts,
                       sw.score    AS score,
                       sw.profile_text AS profile_text
                ORDER BY sw.ts ASC
                """
            )
            return [dict(r) for r in result]

    def upsert_artist(self, artist_id: str, props: dict) -> None:
        """
        Merge or create an Artist node. Only scalar properties are stored
        (lists and dicts are skipped — use separate relationship nodes for those).
        """
        if not self._driver:
            return
        scalar = {
            k: v for k, v in props.items()
            if v is not None and isinstance(v, (str, int, float, bool))
        }
        if not scalar:
            return
        with self._driver.session() as s:
            s.run(
                "MERGE (a:Artist {artist_id: $aid}) SET a += $props",
                aid=artist_id, props=scalar,
            )

    def save_similar_edges(self, artist_id: str, similar_names: list[str], source: str = "lastfm") -> None:
        """Create SIMILAR_TO relationships to named artists (creates Artist stubs if needed)."""
        if not self._driver or not similar_names:
            return
        with self._driver.session() as s:
            for sim_name in similar_names:
                s.run(
                    """
                    MERGE (a:Artist {artist_id: $aid})
                    MERGE (b:Artist {name: $sim_name})
                      ON CREATE SET b.artist_id = $sim_slug
                    MERGE (a)-[:SIMILAR_TO {source: $source}]->(b)
                    """,
                    aid=artist_id, sim_name=sim_name,
                    sim_slug=_slug(sim_name), source=source,
                )

    def get_yes_artist_ids(self) -> list[str]:
        """Return artist_ids that received a YES swipe, ordered by recency."""
        if not self._driver:
            return []
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (a:Artist)-[:RECEIVED_SWIPE]->(sw:Swipe {decision: 'yes'})
                RETURN a.artist_id AS artist_id
                ORDER BY sw.ts DESC
                """
            )
            return [r["artist_id"] for r in result]

    def count_swipes(self) -> dict[str, int]:
        """Return counts by decision."""
        if not self._driver:
            return {}
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH ()-[:RECEIVED_SWIPE]->(sw:Swipe)
                RETURN sw.decision AS decision, count(*) AS n
                """
            )
            return {r["decision"]: r["n"] for r in result}

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with self._driver.session() as s:
            for q in [
                "CREATE CONSTRAINT artist_id IF NOT EXISTS FOR (a:Artist) REQUIRE a.artist_id IS UNIQUE",
                "CREATE CONSTRAINT swipe_id  IF NOT EXISTS FOR (s:Swipe)  REQUIRE s.swipe_id  IS UNIQUE",
            ]:
                try:
                    s.run(q)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — one driver per process
# ─────────────────────────────────────────────────────────────────────────────

_instance: Neo4jClient | None = None


def get_client() -> Neo4jClient:
    global _instance
    if _instance is None:
        _instance = Neo4jClient()
    return _instance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
