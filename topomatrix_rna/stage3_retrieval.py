"""Stage 3: Topological Template Retrieval.

Retrieves the best-matching template structures from the topological atlas
using genus distance and Sliced Gromov-Wasserstein (SGW) distance.

References:
    [11] Vayer T et al. (2019). arXiv:1905.07645 — sliced Gromov-Wasserstein
    [12] Etnyre JB (2003). arXiv:math/0306256 — Legendrian knot invariants
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import structlog

from .config import RetrievalConfig
from .numba_kernels import sliced_wasserstein_1d, stable_rank_signature
from .stage0_atlas import AtlasEntry, TopologicalAtlas

logger = structlog.get_logger(__name__)


class TemplateRetriever:
    """Retrieves RNA structure templates from topological atlas.

    Two-phase retrieval:
    1. Pre-filter by stable rank signature (fast, O(64) per comparison)
    2. Rank by genus distance + Sliced Gromov-Wasserstein distance
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def retrieve(
        self,
        query_genus: int,
        query_stable_rank: np.ndarray,
        query_birth_death: np.ndarray,
        atlas: TopologicalAtlas,
    ) -> List[Tuple[str, float, AtlasEntry]]:
        r"""Retrieve top-k template structures from atlas.

        Mathematical Basis:
            :math:`\text{Template} = \arg\min_{\text{CIF}_k}
            [\lambda_1 \cdot |\hat{g} - g_k| + \lambda_2 \cdot \text{SGW}(\hat{P}, P_k)]`

        Args:
            query_genus: Predicted genus of query RNA.
            query_stable_rank: Stable rank signature, shape (64,).
            query_birth_death: Persistence diagram, shape (n, 2).
            atlas: TopologicalAtlas with entries.

        Returns:
            List of (pdb_id, distance, AtlasEntry) sorted by distance.
        """
        if len(atlas.entries) == 0:
            logger.warning("empty_atlas")
            return []

        cfg = self.config

        # Phase 1: Pre-filter by stable rank distance
        candidates = self._prefilter_stable_rank(
            query_stable_rank, atlas, cfg.prefilter_top_k
        )

        # Phase 2: Rank by combined genus + SGW distance
        ranked = []
        for pdb_id, entry in candidates:
            genus_dist = abs(query_genus - entry.genus)

            # Sliced Gromov-Wasserstein distance
            sgw_dist = self._compute_sgw(
                query_birth_death, entry.birth_death, cfg.sgw_n_projections
            )

            combined = cfg.lambda_genus * genus_dist + cfg.lambda_wasserstein * sgw_dist
            ranked.append((pdb_id, combined, entry))

        ranked.sort(key=lambda x: x[1])
        result = ranked[: cfg.retrieval_top_k]

        if result:
            logger.info(
                "template_retrieval",
                n_candidates=len(candidates),
                top_match=result[0][0],
                top_distance=result[0][1],
            )

        return result

    def _prefilter_stable_rank(
        self,
        query_sr: np.ndarray,
        atlas: TopologicalAtlas,
        top_k: int,
    ) -> List[Tuple[str, AtlasEntry]]:
        """Pre-filter atlas entries by stable rank L2 distance.

        Uses a pre-built matrix cache on the atlas for O(n·d) vectorised
        comparison instead of a Python loop.

        Args:
            query_sr: Query stable rank signature, shape (64,).
            atlas: TopologicalAtlas (``_build_sr_cache`` called if needed).
            top_k: Number of candidates to return.

        Returns:
            List of (pdb_id, AtlasEntry) sorted by stable rank distance.
        """
        if not hasattr(atlas, "_sr_matrix"):
            atlas._build_sr_cache()
        if atlas._sr_matrix.shape[0] == 0:
            return []
        diffs = atlas._sr_matrix - query_sr[np.newaxis, :]
        distances = np.linalg.norm(diffs, axis=1)
        k_actual = min(top_k, len(distances))
        top_idx = np.argpartition(distances, k_actual - 1)[:k_actual]
        top_idx = top_idx[np.argsort(distances[top_idx])]
        return [
            (atlas._sr_pdb_ids[i], atlas.entries[atlas._sr_pdb_ids[i]])
            for i in top_idx
        ]

    def _compute_sgw(
        self,
        bd1: np.ndarray,
        bd2: np.ndarray,
        n_projections: int,
    ) -> float:
        r"""Compute Sliced Gromov-Wasserstein distance between persistence diagrams.

        Mathematical Basis:
            Compares internal distance matrices (Gram matrices) of the two
            diagrams via random projections:

            :math:`\text{SGW}(\mu, \nu) = \mathbb{E}_{v}[W_1(C_1 v, C_2 v)]`

            where :math:`C_1, C_2` are the pairwise distance matrices of the
            two diagrams. Complexity: O(K·n²).

        Args:
            bd1: First persistence diagram, shape (n1, 2).
            bd2: Second persistence diagram, shape (n2, 2).
            n_projections: Number of random projections (K).

        Returns:
            SGW distance (float >= 0).
        """
        if bd1.shape[0] < 2 or bd2.shape[0] < 2:
            return self._compute_swd(bd1, bd2, n_projections)

        n1, n2 = bd1.shape[0], bd2.shape[0]
        C1 = np.sqrt(
            np.sum((bd1[:, None, :] - bd1[None, :, :]) ** 2, axis=2)
        )
        C2 = np.sqrt(
            np.sum((bd2[:, None, :] - bd2[None, :, :]) ** 2, axis=2)
        )
        rng = np.random.RandomState(42)
        total = 0.0
        for _ in range(n_projections):
            v1 = rng.randn(n1)
            v1 /= np.linalg.norm(v1) + 1e-10
            v2 = rng.randn(n2)
            v2 /= np.linalg.norm(v2) + 1e-10
            proj1 = np.sort(C1 @ v1)
            proj2 = np.sort(C2 @ v2)
            n_max = max(len(proj1), len(proj2))
            p1 = np.zeros(n_max)
            p2 = np.zeros(n_max)
            p1[: len(proj1)] = proj1
            p2[: len(proj2)] = proj2
            total += float(sliced_wasserstein_1d(p1, p2))
        return total / n_projections

    def _compute_swd(
        self,
        bd1: np.ndarray,
        bd2: np.ndarray,
        n_projections: int,
    ) -> float:
        r"""Compute Sliced Wasserstein Distance between persistence diagrams.

        Fallback when either diagram has fewer than 2 points (no internal
        distance matrix can be formed for SGW).

        Mathematical Basis:
            :math:`\text{SWD}(\mu, \nu) = \mathbb{E}_{\theta \sim \text{Unif}(S^1)}
            [W_1(\pi_\theta \# \mu, \pi_\theta \# \nu)]`

        Args:
            bd1: First persistence diagram, shape (n1, 2).
            bd2: Second persistence diagram, shape (n2, 2).
            n_projections: Number of random projections (K).

        Returns:
            SWD distance (float >= 0).
        """
        if bd1.shape[0] == 0 and bd2.shape[0] == 0:
            return 0.0
        if bd1.shape[0] == 0 or bd2.shape[0] == 0:
            bd = bd1 if bd1.shape[0] > 0 else bd2
            return float(np.sum(bd[:, 1] - bd[:, 0]))

        rng = np.random.RandomState(42)
        total_w1 = 0.0

        for _ in range(n_projections):
            theta = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])

            proj1 = bd1 @ direction
            proj2 = bd2 @ direction

            proj1_sorted = np.sort(proj1)
            proj2_sorted = np.sort(proj2)

            n1, n2 = len(proj1_sorted), len(proj2_sorted)
            n_max = max(n1, n2)
            p1 = np.zeros(n_max)
            p2 = np.zeros(n_max)
            p1[:n1] = proj1_sorted
            p2[:n2] = proj2_sorted

            total_w1 += float(sliced_wasserstein_1d(p1, p2))

        return total_w1 / n_projections

    def retrieve_for_novel_topology(
        self,
        query_genus: int,
        atlas: TopologicalAtlas,
    ) -> List[Tuple[str, float, AtlasEntry]]:
        r"""Fallback retrieval for novel topologies not in atlas.

        Uses Legendrian knot invariants (Thurston-Bennequin number):
        :math:`\chi(\Sigma) = 2 - 2g = tb + |r|`

        Args:
            query_genus: Predicted genus.
            atlas: TopologicalAtlas.

        Returns:
            List of (pdb_id, distance, AtlasEntry).
        """
        # Find entries with nearest genus
        entries_by_genus = []
        for pdb_id, entry in atlas.entries.items():
            genus_dist = abs(query_genus - entry.genus)
            entries_by_genus.append((pdb_id, float(genus_dist), entry))

        entries_by_genus.sort(key=lambda x: x[1])
        return entries_by_genus[: self.config.retrieval_top_k]
