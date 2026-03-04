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
        self._saddle_merges: dict = {}  # saddle_idx -> (min_of_component_a, min_of_component_b)

    def build_from_energy(
        self, energies: np.ndarray, adjacency: np.ndarray
    ) -> None:
        r"""Build Reeb graph from energy function on RNA conformations.

        Mathematical Basis:
            :math:`\text{Reeb}(f) = X / \sim`, where
            :math:`x \sim y \iff f(x) = f(y)` and same connected component
            of :math:`f^{-1}(f(x))`.

            Component merging uses path-compressed union-find (O(α) per
            operation) instead of an O(n) full-array scan.

        Args:
            energies: Energy values per conformation, shape (n,).
            adjacency: Adjacency matrix, shape (n, n).
        """
        n = energies.shape[0]
        if n == 0:
            return

        # Union-Find with path compression and union by rank
        parent = np.arange(n, dtype=np.int64)
        rank_uf = np.zeros(n, dtype=np.int64)
        # comp_min maps each uf-root → index of the lowest-energy minimum
        # in that component.  Maintained through every union() call so that
        # _saddle_merges stores actual minimum indices, not arbitrary roots.
        comp_min: dict = {}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path halving
                x = parent[x]
            return x

        def union(a: int, b: int) -> int:
            ra, rb = find(a), find(b)
            if ra == rb:
                return ra
            if rank_uf[ra] < rank_uf[rb]:
                parent[ra] = rb
                new_root = rb
            elif rank_uf[ra] > rank_uf[rb]:
                parent[rb] = ra
                new_root = ra
            else:
                parent[rb] = ra
                rank_uf[ra] += 1
                new_root = ra
            # Propagate minimum representative to new root
            min_a = comp_min.get(ra, ra)
            min_b = comp_min.get(rb, rb)
            comp_min[new_root] = (
                min_a if energies[min_a] <= energies[min_b] else min_b
            )
            return new_root

        sorted_idx = np.argsort(energies)
        self.nodes = []
        self.edges = []
        self._saddle_merges = {}
        processed = np.zeros(n, dtype=bool)

        for idx in sorted_idx:
            neighbors_done = [
                j for j in range(n)
                if adjacency[idx, j] > 0 and processed[j]
            ]

            if not neighbors_done:
                # Birth of new component (minimum)
                self.nodes.append(ReebNode(idx, float(energies[idx]), "min"))
                comp_min[idx] = idx  # This node IS its own minimum representative
            else:
                roots = list({find(j) for j in neighbors_done})
                ntype = "regular" if len(roots) == 1 else "saddle"
                self.nodes.append(ReebNode(idx, float(energies[idx]), ntype))
                if len(roots) >= 2:
                    # Record minimum representatives (not uf roots) for Elder Rule.
                    # comp_min[root] is the lowest-energy minimum in that component.
                    self._saddle_merges[idx] = (
                        comp_min.get(roots[0], roots[0]),
                        comp_min.get(roots[1], roots[1]),
                    )
                for c in roots[1:]:
                    self.edges.append((idx, c))
                    union(roots[0], c)

            processed[idx] = True

        # Detect local maxima
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

        Uses the Elder Rule (Cohen-Steiner 2006): each minimum is paired
        with the lowest-energy saddle on the path in the Reeb graph from
        that minimum to the nearest deeper minimum (its cancellation partner).

        The global minimum is assigned infinite persistence.

        Mathematical Basis:
            :math:`\text{persistence}(\text{basin}_i) =
            f(\text{saddle}_{i \to \text{deeper}}) - f(\text{min}_i)`

        Args:
            n_basins: Number of basins to return.

        Returns:
            List of (min_index, persistence) tuples, most persistent first.
        """
        minima = [n for n in self.nodes if n.node_type == "min"]
        saddle_nodes = {n.index: n for n in self.nodes if n.node_type == "saddle"}

        if len(minima) == 0:
            return []

        # Sort minima by energy (ascending = deepest first)
        minima_sorted = sorted(minima, key=lambda m: m.value)
        global_min = minima_sorted[0]

        # Energy lookup by node index — used for _saddle_merges comparison
        energy_by_idx = {n.index: n.value for n in self.nodes}

        basins: List[Tuple[int, float]] = []
        for m in minima:
            if m.index == global_min.index:
                persistence = 1e6  # Global minimum — infinite persistence
            else:
                best_saddle_val = float("inf")

                if self._saddle_merges:
                    # Elder Rule: _saddle_merges stores actual minimum indices,
                    # so we can use direct index equality to identify which side
                    # of a saddle belongs to minimum m's component.
                    for s_idx, (min_a_idx, min_b_idx) in self._saddle_merges.items():
                        s_node = saddle_nodes.get(s_idx)
                        if s_node is None or s_node.value <= m.value:
                            continue  # Saddle must be strictly above m

                        a_is_m   = (min_a_idx == m.index)
                        b_is_m   = (min_b_idx == m.index)
                        a_deeper = (
                            energy_by_idx.get(min_a_idx, float("inf")) < m.value
                            and not a_is_m
                        )
                        b_deeper = (
                            energy_by_idx.get(min_b_idx, float("inf")) < m.value
                            and not b_is_m
                        )

                        if (a_is_m and b_deeper) or (b_is_m and a_deeper):
                            if s_node.value < best_saddle_val:
                                best_saddle_val = s_node.value

                # Fallback: if no topology-aware saddle found (e.g. isolated
                # minimum with no deeper basin accessible), use the lowest saddle
                # above m that connects to any deeper minimum.
                if best_saddle_val == float("inf"):
                    for s_node in saddle_nodes.values():
                        if s_node.value > m.value and s_node.value < best_saddle_val:
                            # Check that there exists at least one deeper minimum
                            if any(
                                m2.value < m.value
                                for m2 in minima
                                if m2.index != m.index
                            ):
                                best_saddle_val = s_node.value

                persistence = (
                    best_saddle_val - m.value
                    if best_saddle_val < float("inf")
                    else 0.0
                )

            basins.append((m.index, persistence))

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
