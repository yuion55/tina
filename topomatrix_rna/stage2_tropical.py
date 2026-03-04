"""Stage 2: Hierarchical Tropical Geometry (Basin Census).

Implements tropical semiring optimization for RNA secondary structure
and basin enumeration via Newton polytope decomposition.

References:
    [8] Speyer D, Sturmfels B (2004). Not Am Math Soc 51:1145-1156
    [9] Pachter L, Sturmfels B (2004). Proc Natl Acad Sci 101:16138-16143
    [10] Lyngso RB, Pedersen CNS (2000). J Comput Biol 7:409-427
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import structlog
from scipy import sparse

from .config import TropicalConfig, RNABiologyConstants
from .data_utils import encode_sequence
from .numba_kernels import tropical_gaussian_elim, tropical_min_plus

logger = structlog.get_logger(__name__)


class TropicalBasinCensus:
    """Enumerates RNA folding basins using tropical geometry.

    Uses tropical semiring (min, +) to find optimal and sub-optimal
    secondary structures as vertices of the Newton polytope.
    """

    def __init__(self, config: TropicalConfig, bio_config: Optional[RNABiologyConstants] = None) -> None:
        self.config = config
        self._bio_config = bio_config or RNABiologyConstants()
        self._c4prime_coords: Optional[np.ndarray] = None

    def set_backbone_coords(self, c4prime_coords: np.ndarray) -> None:
        """Set C4' coordinates for Mg2+ electrostatic penalty computation."""
        self._c4prime_coords = c4prime_coords

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

        Vectorised: replaces O(L²) Python loop with NumPy broadcasting (~1000× faster).

        Args:
            encoded: Encoded sequence, shape (L,).
            contact_map: Contact probabilities, shape (L, L).

        Returns:
            Weight matrix, shape (L, L). Lower = more favorable.
        """
        L = encoded.shape[0]
        W = np.full((L, L), np.inf)

        # Convert sparse to dense once
        if sparse.issparse(contact_map):
            cm = contact_map.toarray()
        else:
            cm = np.asarray(contact_map)

        # Build bp_weights matrix via vectorised Watson-Crick scoring
        # Encoding: A=0, C=1, G=2, U=3
        bp_weights = np.full((L, L), np.inf)
        pair_rules = [
            (0, 3, 0.9),   # A-U
            (3, 0, 0.9),   # U-A
            (1, 2, 1.0),   # C-G
            (2, 1, 1.0),   # G-C
            (2, 3, 0.7),   # G-U wobble
            (3, 2, 0.7),   # U-G wobble
        ]
        for (si, sj, wt) in pair_rules:
            mask = (encoded[:, None] == si) & (encoded[None, :] == sj)
            bp_weights[mask] = self.config.weight_bp * wt

        # Minimum loop length = 4: only upper triangle with k >= 4
        loop_mask = np.zeros((L, L), dtype=bool)
        if L > 4:
            rows, cols = np.triu_indices(L, k=4)
            loop_mask[rows, cols] = True

        # Valid positions: finite bp_weight AND loop constraint
        valid = loop_mask & np.isfinite(bp_weights)

        # Apply contact probability scaling
        W[valid] = bp_weights[valid] * (1.0 + cm[valid])

        # Stacking bonus: position (i,j) gets bonus if (i-1, j+1) is also valid
        # i.e. shift valid mask by (+1 row, -1 col) = check W[i-1, j+1] < inf
        if L > 5:
            # W[i,j] += weight_stack  iff  i>0, j<L-1, and (i-1,j+1) is valid
            has_stack = np.zeros((L, L), dtype=bool)
            has_stack[1:, :L-1] = valid[1:, :L-1] & valid[:L-1, 1:]
            # Only apply stacking bonus to cells that are themselves valid
            stack_apply = valid & has_stack
            W[stack_apply] += self.config.weight_stack

        # Apply electrostatic penalty if C4' coordinates are available
        if self._c4prime_coords is not None:
            ep = compute_electrostatic_penalty(self._c4prime_coords, self._bio_config)
            W[valid] += ep[valid]

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


def compute_electrostatic_penalty(
    coords_c4prime: np.ndarray,
    config: RNABiologyConstants,
    mg_conc_mm: float = 2.0,
) -> np.ndarray:
    """Debye-Hückel electrostatic penalty matrix.

    Returns (L, L) penalty array. Penalty is positive (unfavourable) for
    close phosphate pairs that are not base-paired — model of backbone
    repulsion screened by Mg2+.

    Debye length λ ≈ 7.0 Å at 2 mM Mg2+.
    Penalty = exp(-r_ij / λ) for all pairs where r < 15 Å and |i-j| > 3.

    Source: Draper DE (2004) RNA 10:335-343.

    Args:
        coords_c4prime: C4' atom coordinates as backbone proxy, shape (L, 3).
        config: RNABiologyConstants instance.
        mg_conc_mm: Mg2+ concentration in mM (default 2.0).

    Returns:
        Penalty matrix, shape (L, L).
    """
    DEBYE_LAMBDA = config.debye_length_mg_2mm  # 7.0 Å
    CUTOFF = 15.0   # Å, beyond this the penalty is negligible
    SCALE = 0.05    # kcal/mol units, calibrated against Turner params

    L = coords_c4prime.shape[0]
    penalty = np.zeros((L, L), dtype=np.float32)

    diff = coords_c4prime[:, None, :] - coords_c4prime[None, :, :]  # (L,L,3)
    dist = np.linalg.norm(diff, axis=-1)  # (L,L)

    close_mask = (dist < CUTOFF) & (dist > 0.1)
    seq_mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) > 3
    active = close_mask & seq_mask

    penalty[active] = SCALE * np.exp(-dist[active] / DEBYE_LAMBDA)
    return penalty


def compute_pseudotorsions(
    c4prime_coords: np.ndarray,
    p_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute η and θ pseudotorsion angles for each nucleotide.

    η(i) = torsion(C4'(i-1), P(i),  C4'(i), P(i+1))
    θ(i) = torsion(P(i),     C4'(i), P(i+1), C4'(i+1))

    Source: Wadley LM, Pyle AM (2004) NAR 32:6650-6659.

    Args:
        c4prime_coords: C4' atom coordinates, shape (L, 3).
        p_coords: Phosphorus atom coordinates, shape (L, 3).

    Returns:
        Tuple of (eta, theta) each shape (L,) in degrees. NaN at termini.
    """
    L = c4prime_coords.shape[0]
    eta = np.full(L, np.nan)
    theta = np.full(L, np.nan)

    if L < 3:
        return eta, theta

    # Vectorised torsion computation for all interior residues
    # η(i): torsion(C4'(i-1), P(i), C4'(i), P(i+1)) for i in [1, L-2]
    b1_eta = p_coords[1:L-1] - c4prime_coords[0:L-2]         # (L-2, 3)
    b2_eta = c4prime_coords[1:L-1] - p_coords[1:L-1]          # (L-2, 3)
    b3_eta = p_coords[2:L] - c4prime_coords[1:L-1]            # (L-2, 3)
    eta[1:L-1] = _vectorised_torsion(b1_eta, b2_eta, b3_eta)

    # θ(i): torsion(P(i), C4'(i), P(i+1), C4'(i+1)) for i in [1, L-2]
    b1_theta = c4prime_coords[1:L-1] - p_coords[1:L-1]        # (L-2, 3)
    b2_theta = p_coords[2:L] - c4prime_coords[1:L-1]           # (L-2, 3)
    b3_theta = c4prime_coords[2:L] - p_coords[2:L]             # (L-2, 3)
    theta[1:L-1] = _vectorised_torsion(b1_theta, b2_theta, b3_theta)

    return eta, theta


def _vectorised_torsion(
    b1: np.ndarray, b2: np.ndarray, b3: np.ndarray
) -> np.ndarray:
    """Vectorised torsion angle computation for N bond triplets.

    Args:
        b1, b2, b3: Bond vectors, each shape (N, 3).

    Returns:
        Torsion angles in degrees, shape (N,).
    """
    n1 = np.cross(b1, b2)                                      # (N, 3)
    n2 = np.cross(b2, b3)                                      # (N, 3)
    b2_norm = np.linalg.norm(b2, axis=1, keepdims=True) + 1e-12
    m1 = np.cross(n1, b2 / b2_norm)                            # (N, 3)
    x = np.einsum('ij,ij->i', n1, n2)                          # (N,)
    y = np.einsum('ij,ij->i', m1, n2)                          # (N,)
    return np.degrees(np.arctan2(y, x))
