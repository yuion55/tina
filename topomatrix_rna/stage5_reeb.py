"""Stage 5: Reeb Graph Basin Enumeration.

Constructs Reeb graphs from energy landscapes to enumerate topologically
distinct folding basins via extended persistence pairing.

References:
    [18] Forman R (2002). Séminaires & Congrès 7:135-190 — discrete Morse theory
    [19] Cohen-Steiner D et al. (2006). SCG'06 360-369 — extended persistence
    [20] Flamm C et al. (2000). RNA 6:325-338 — RNA barrier trees
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ReebNode:
    """Node in a Reeb graph."""

    def __init__(self, index: int, value: float, node_type: str = "regular") -> None:
        self.index = index
        self.value = value
        self.node_type = node_type  # "min", "max", "saddle", "regular"
        self.neighbors: List[int] = []


class ReebGraph:
    """Reeb graph for energy landscape analysis.

    Constructed from a scalar function on a simplicial complex.
    Used to enumerate folding basins via persistent homology.
    """

    def __init__(self) -> None:
        self.nodes: List[ReebNode] = []
        self.edges: List[Tuple[int, int]] = []

    def build_from_energy(
        self, energies: np.ndarray, adjacency: np.ndarray
    ) -> None:
        r"""Build Reeb graph from energy function on RNA conformations.

        Mathematical Basis:
            :math:`\text{Reeb}(f) = X / \sim`, where
            :math:`x \sim y \iff f(x) = f(y)` and same connected component
            of :math:`f^{-1}(f(x))`.

        Args:
            energies: Energy values per conformation, shape (n,).
            adjacency: Adjacency matrix, shape (n, n).
        """
        n = energies.shape[0]
        if n == 0:
            return

        # Sort vertices by energy
        sorted_idx = np.argsort(energies)

        # Build Reeb graph via sweep
        self.nodes = []
        self.edges = []
        component = np.full(n, -1, dtype=np.int64)
        n_components = 0

        for idx in sorted_idx:
            # Find neighboring vertices already processed
            neighbors_processed = []
            for j in range(n):
                if adjacency[idx, j] > 0 and component[j] >= 0:
                    neighbors_processed.append(j)

            if len(neighbors_processed) == 0:
                # Birth of new component (minimum)
                component[idx] = n_components
                self.nodes.append(
                    ReebNode(idx, float(energies[idx]), "min")
                )
                n_components += 1

            else:
                # Find distinct components among neighbors
                neighbor_components = set()
                for j in neighbors_processed:
                    neighbor_components.add(component[j])

                comp_list = list(neighbor_components)

                if len(comp_list) == 1:
                    # Regular vertex or local max
                    component[idx] = comp_list[0]
                    self.nodes.append(
                        ReebNode(idx, float(energies[idx]), "regular")
                    )
                else:
                    # Saddle point — merge components
                    merged_comp = comp_list[0]
                    component[idx] = merged_comp

                    for c in comp_list[1:]:
                        for k in range(n):
                            if component[k] == c:
                                component[k] = merged_comp
                        # Add edge representing merge
                        self.edges.append((idx, c))

                    self.nodes.append(
                        ReebNode(idx, float(energies[idx]), "saddle")
                    )

        # Detect local maxima (processed last among their neighbors)
        for node in self.nodes:
            if node.node_type == "regular":
                idx = node.index
                is_max = True
                for j in range(n):
                    if adjacency[idx, j] > 0 and energies[j] > energies[idx]:
                        is_max = False
                        break
                if is_max:
                    node.node_type = "max"

    def get_basins(self, n_basins: int = 5) -> List[Tuple[int, float]]:
        r"""Extract topologically distinct basins ranked by persistence.

        Mathematical Basis:
            :math:`\text{persistence}(\text{basin}_i) =
            |f(\text{min}_i) - f(\text{saddle}_i)|`

        Args:
            n_basins: Number of basins to return.

        Returns:
            List of (min_index, persistence) tuples.
        """
        minima = [n for n in self.nodes if n.node_type == "min"]
        saddles = [n for n in self.nodes if n.node_type == "saddle"]

        if len(minima) == 0:
            return []

        # Pair minima with saddles
        basins = []
        for m in minima:
            # Find nearest saddle (by index proximity or energy)
            best_saddle_val = float("inf")
            for s in saddles:
                if s.value > m.value and s.value < best_saddle_val:
                    best_saddle_val = s.value

            if best_saddle_val == float("inf"):
                # Global minimum — infinite persistence (use large value)
                persistence = 1e6
            else:
                persistence = best_saddle_val - m.value

            basins.append((m.index, persistence))

        # Sort by persistence (most stable first)
        basins.sort(key=lambda x: -x[1])
        return basins[:n_basins]


class ReebBasinEnumerator:
    """Enumerates RNA folding basins using Reeb graph analysis."""

    def __init__(self, n_basins: int = 5) -> None:
        self.n_basins = n_basins

    def enumerate(
        self,
        candidate_energies: np.ndarray,
        candidate_coords: List[np.ndarray],
    ) -> List[int]:
        """Select topologically distinct basins from candidates.

        Args:
            candidate_energies: Energy of each candidate, shape (n,).
            candidate_coords: List of coordinate arrays.

        Returns:
            Indices of selected basin representatives.
        """
        n = len(candidate_energies)
        if n <= self.n_basins:
            return list(range(n))

        # Build adjacency from coordinate similarity
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                L = min(candidate_coords[i].shape[0], candidate_coords[j].shape[0])
                if L > 0:
                    diff = candidate_coords[i][:L] - candidate_coords[j][:L]
                    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                    if rmsd < 20.0:  # Within structural similarity threshold
                        adjacency[i, j] = 1.0
                        adjacency[j, i] = 1.0

        # Build Reeb graph
        reeb = ReebGraph()
        reeb.build_from_energy(candidate_energies, adjacency)

        # Get basins
        basins = reeb.get_basins(self.n_basins)
        if basins:
            return [b[0] for b in basins]
        else:
            # Fallback: select by energy
            sorted_idx = np.argsort(candidate_energies)
            return list(sorted_idx[: self.n_basins])

    def check_morse_inequalities(
        self,
        reeb: ReebGraph,
        betti_numbers: Tuple[int, ...],
    ) -> bool:
        r"""Verify discrete Morse theory consistency.

        Mathematical Basis:
            Morse inequalities: :math:`c_k \geq \beta_k`
            where :math:`c_k` = number of critical k-cells.

        Args:
            reeb: Reeb graph.
            betti_numbers: Tuple of Betti numbers (β₀, β₁, ...).

        Returns:
            True if Morse inequalities are satisfied.
        """
        c0 = sum(1 for n in reeb.nodes if n.node_type == "min")
        c1 = sum(1 for n in reeb.nodes if n.node_type == "saddle")
        c2 = sum(1 for n in reeb.nodes if n.node_type == "max")

        critical_cells = [c0, c1, c2]

        for k, beta_k in enumerate(betti_numbers):
            if k < len(critical_cells) and critical_cells[k] < beta_k:
                logger.warning(
                    "morse_inequality_violated",
                    k=k, c_k=critical_cells[k], beta_k=beta_k,
                )
                return False
        return True
