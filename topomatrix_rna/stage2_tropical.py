"""Stage 2: Hierarchical Tropical Geometry (Basin Census).

Implements tropical semiring optimization for RNA secondary structure
and basin enumeration via Newton polytope decomposition.

References:
    [8] Speyer D, Sturmfels B (2004). Not Am Math Soc 51:1145-1156
    [9] Pachter L, Sturmfels B (2004). Proc Natl Acad Sci 101:16138-16143
    [10] Lyngso RB, Pedersen CNS (2000). J Comput Biol 7:409-427
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import structlog
from scipy import sparse

from .config import TropicalConfig
from .data_utils import encode_sequence
from .numba_kernels import tropical_gaussian_elim, tropical_min_plus

logger = structlog.get_logger(__name__)


class TropicalBasinCensus:
    """Enumerates RNA folding basins using tropical geometry.

    Uses tropical semiring (min, +) to find optimal and sub-optimal
    secondary structures as vertices of the Newton polytope.
    """

    def __init__(self, config: TropicalConfig) -> None:
        self.config = config

    def find_basins(
        self, sequence: str, contact_map: np.ndarray, n_basins: int = 0
    ) -> List[np.ndarray]:
        r"""Find top-k folding basins via tropical optimization.

        Mathematical Basis:
            Free energy as tropical polynomial:
            :math:`F_{\text{trop}}(S) = \bigoplus_{S} \bigotimes_{(i,j) \in S} w_{ij}
            = \min_S \sum_{(i,j) \in S} w_{ij}`

            Recursive interval DP:
            :math:`\text{OPT}(i,j) = \min(w_{ij} + \text{OPT}(i+1,j-1),\;
            \min_{i<k<j}[\text{OPT}(i,k) + \text{OPT}(k+1,j)])`

            Sub-optimal basins are found by systematic perturbation via
            tropical Gaussian elimination and min-plus matrix products,
            replacing random noise with algebraically grounded shifts.

        Args:
            sequence: RNA sequence string.
            contact_map: Contact probability matrix from Stage 1.
            n_basins: Number of basins to find (0 = use config default).

        Returns:
            List of base pair arrays, each shape (n_pairs, 2).
        """
        if n_basins <= 0:
            n_basins = self.config.max_basins

        L = len(sequence)
        if L < 5:
            return [np.empty((0, 2), dtype=np.int64)]

        encoded = encode_sequence(sequence)

        # Compute tropical weight matrix from contact probabilities
        W = self._compute_weight_matrix(encoded, contact_map)

        # Find optimal structure via tropical DP
        basins = []
        opt_pairs = self._tropical_dp(W, L)
        if len(opt_pairs) > 0:
            basins.append(np.array(opt_pairs, dtype=np.int64))

        # Build tropical linear system for systematic sub-optimal generation
        A, b = self._build_tropical_system(W, L)

        for k in range(1, n_basins):
            # Modify right-hand side to steer toward a different solution
            b_k = b.copy()
            if L > 0:
                b_k[k % L] += float(k) * 0.5

            # Solve tropical system to get a perturbation vector
            x_k = tropical_gaussian_elim(A, b_k)

            # Build perturbation diagonal matrix and propagate via min-plus
            n = min(L, x_k.shape[0])
            D_k = np.full((n, n), np.inf)
            for ii in range(n):
                if x_k[ii] < np.inf:
                    D_k[ii, ii] = x_k[ii]

            W_mod = tropical_min_plus(W[:n, :n], D_k)
            # Restore original inf entries where W was inf
            # tropical_min_plus propagates np.inf correctly; no fixup needed.

            pairs = self._tropical_dp(W_mod, L)
            if len(pairs) > 0:
                basins.append(np.array(pairs, dtype=np.int64))

        if len(basins) == 0:
            basins.append(np.empty((0, 2), dtype=np.int64))

        logger.info("tropical_basins_found", n_basins=len(basins), L=L)
        return basins

    def _build_tropical_system(
        self, W: np.ndarray, L: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build tropical linear system A, b from the weight matrix.

        The coefficient matrix is W itself; the right-hand side b[i] is the
        minimum finite weight available for each position (i.e., the cost of
        the best base-pair partner for residue i).

        Args:
            W: Tropical weight matrix, shape (L, L).
            L: Sequence length.

        Returns:
            Tuple of (A, b) where A has shape (L, L) and b has shape (L,).
        """
        A = W.copy()
        b = np.full(L, np.inf)
        for i in range(L):
            row_min = np.min(W[i])
            b[i] = row_min if row_min < np.inf else 0.0
        return A, b

    def _compute_weight_matrix(
        self, encoded: np.ndarray, contact_map: np.ndarray
    ) -> np.ndarray:
        """Compute tropical weight matrix from sequence and contact probabilities.

        Weights combine base-pairing affinity with contact probability.
        Handles both dense ``np.ndarray`` and sparse ``csr_matrix`` contact maps.

        Args:
            encoded: Encoded sequence, shape (L,).
            contact_map: Contact probabilities, shape (L, L).

        Returns:
            Weight matrix, shape (L, L). Lower = more favorable.
        """
        L = encoded.shape[0]
        W = np.full((L, L), np.inf)

        # Convert sparse to dense for indexing within the DP loop
        if sparse.issparse(contact_map):
            cm = contact_map.toarray()
        else:
            cm = contact_map

        for i in range(L):
            for j in range(i + 4, L):  # Minimum loop length = 4
                si, sj = encoded[i], encoded[j]

                # Base pair weight
                bp_weight = np.inf
                if (si == 0 and sj == 3) or (si == 3 and sj == 0):
                    bp_weight = self.config.weight_bp  # A-U
                elif (si == 1 and sj == 2) or (si == 2 and sj == 1):
                    bp_weight = self.config.weight_bp * 1.5  # G-C (stronger)
                elif (si == 2 and sj == 3) or (si == 3 and sj == 2):
                    bp_weight = self.config.weight_bp * 0.5  # G-U wobble

                if bp_weight < np.inf:
                    # Combine with contact probability
                    p_contact = cm[i, j] if cm.shape[0] > 0 else 0.5
                    W[i, j] = bp_weight * (1.0 + p_contact)

                    # Stacking bonus for consecutive base pairs
                    if (i > 0 and j < L - 1 and
                            W[i - 1, j + 1] < np.inf):
                        W[i, j] += self.config.weight_stack

        return W

    def _tropical_dp(
        self, W: np.ndarray, L: int
    ) -> List[Tuple[int, int]]:
        """Tropical dynamic programming for optimal base pairing.

        Uses Nussinov-style DP in the tropical semiring.

        Args:
            W: Weight matrix, shape (L, L).
            L: Sequence length.

        Returns:
            List of (i, j) base pairs.
        """
        if L < 5:
            return []

        # DP table: OPT[i][j] = min energy for subsequence [i..j]
        OPT = np.zeros((L, L))
        traceback = np.full((L, L), -1, dtype=np.int64)

        # Fill DP table
        for span in range(5, L):
            for i in range(L - span):
                j = i + span

                # Option 1: j unpaired
                best = OPT[i, j - 1]
                best_k = -1

                # Option 2: j pairs with some k
                for k in range(i, j - 3):
                    if W[k, j] < np.inf:
                        val = W[k, j]
                        if k > i:
                            val += OPT[i, k - 1]
                        if k + 1 < j - 1:
                            val += OPT[k + 1, j - 1]
                        if val < best:
                            best = val
                            best_k = k

                # Option 3: bifurcation
                for k in range(i + 1, j - 1):
                    val = OPT[i, k] + OPT[k + 1, j]
                    if val < best:
                        best = val
                        best_k = -(k + 2)  # Encode bifurcation point

                OPT[i, j] = best
                traceback[i, j] = best_k

        # Traceback to recover base pairs
        pairs: List[Tuple[int, int]] = []
        self._traceback(traceback, W, 0, L - 1, pairs)
        return pairs

    def _traceback(
        self,
        tb: np.ndarray,
        W: np.ndarray,
        i_start: int,
        j_start: int,
        pairs: List[Tuple[int, int]],
    ) -> None:
        """Iterative traceback through DP table (avoids stack overflow at L > 800).

        Replaces the recursive version with an explicit stack to support
        arbitrarily long sequences without hitting Python's recursion limit.

        Args:
            tb: Traceback table, shape (L, L).
            W: Weight matrix, shape (L, L).
            i_start: Start of the interval.
            j_start: End of the interval.
            pairs: Output list of (i, j) base pairs (modified in place).
        """
        stack = [(i_start, j_start)]
        while stack:
            i, j = stack.pop()
            if i >= j - 3:
                continue
            k = int(tb[i, j])
            if k == -1:
                # j is unpaired
                stack.append((i, j - 1))
            elif k >= 0:
                # k pairs with j
                pairs.append((k, j))
                if k > i:
                    stack.append((i, k - 1))
                if k + 1 < j - 1:
                    stack.append((k + 1, j - 1))
            else:
                # Bifurcation at -(k+2)
                split = -(k + 2)
                stack.append((i, split))
                stack.append((split + 1, j))

    def basins_to_coordinates(
        self,
        sequence: str,
        basins: List[np.ndarray],
        template_coords: np.ndarray,
    ) -> List[np.ndarray]:
        """Convert basin base-pair sets to initial 3D coordinate guesses.

        Uses template coordinates as base and adjusts for each basin's
        specific base-pairing pattern.

        Args:
            sequence: RNA sequence.
            basins: List of base pair arrays from find_basins.
            template_coords: Template C3' coordinates, shape (L, 3).

        Returns:
            List of coordinate arrays, each shape (L, 3).
        """
        L = len(sequence)
        coords_list = []

        for basin_pairs in basins:
            coords = template_coords[:L].copy() if template_coords.shape[0] >= L else \
                np.zeros((L, 3))

            # Adjust coordinates based on base pairing
            if basin_pairs.shape[0] > 0:
                for pair_idx in range(basin_pairs.shape[0]):
                    pi, pj = basin_pairs[pair_idx, 0], basin_pairs[pair_idx, 1]
                    if pi < L and pj < L:
                        # Move paired residues closer (target ~6.5Å C3'-C3')
                        midpoint = (coords[pi] + coords[pj]) / 2.0
                        direction = coords[pj] - coords[pi]
                        dist = np.linalg.norm(direction)
                        if dist > 0:
                            target_dist = 6.5
                            coords[pi] = midpoint - direction / dist * target_dist / 2
                            coords[pj] = midpoint + direction / dist * target_dist / 2

            coords_list.append(coords)

        return coords_list
