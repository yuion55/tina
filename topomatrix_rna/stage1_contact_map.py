"""Stage 1: Contact Map via Renormalization Group Matrix Field Theory.

Computes contact probability maps using RG blocking and matrix model
saddle-point equations. For sequences longer than a threshold, uses
functional renormalization group (FRG) flow.

References:
    [5] Orland H, Zee A (2002). Nucl Phys B 620:456-476 — matrix field theory RNA
    [6] Berges J et al. (2002). Phys Rep 363:223-386 — functional RG review
    [7] Litim DF (2001). Phys Rev D 64:105007 — optimal FRG regulator
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import structlog
from scipy import sparse

from .config import ContactMapConfig, RNABiologyConstants
from .data_utils import encode_sequence
from .numba_kernels import compute_genus_gauss_code, rg_block_contact_map

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Leontis-Westhof 12-family weights
# Key: (base_i, base_j, family_abbr) where family_abbr is one of:
# cWW, tWW, cWH, tWH, cWS, tWS, cHH, tHH, cHS, tHS, cSS, tSS
# Source: Leontis NB, Westhof E (2001) RNA 7:499-512
# ---------------------------------------------------------------------------
LW_BP_WEIGHTS: dict[tuple[str, str, str], float] = {
    ('G', 'C', 'cWW'): 1.0,
    ('C', 'G', 'cWW'): 1.0,
    ('A', 'U', 'cWW'): 0.9,
    ('U', 'A', 'cWW'): 0.9,
    ('G', 'U', 'cWW'): 0.7,   # wobble
    ('U', 'G', 'cWW'): 0.7,   # wobble
    ('G', 'A', 'tHS'): 0.5,   # kink-turn G•A
    ('A', 'G', 'tHS'): 0.5,
    ('G', 'A', 'tWS'): 0.45,
    ('A', 'G', 'tWS'): 0.45,
    ('G', 'G', 'cSS'): 0.4,
    ('A', 'A', 'cWS'): 0.4,
    ('A', 'C', 'cWS'): 0.35,
    ('C', 'A', 'cWS'): 0.35,
}


def get_lw_weight(base_i: str, base_j: str, family: str = 'cWW') -> float:
    """Return Leontis-Westhof weight for a base pair.

    Falls back symmetrically, then to default non-canonical weight.

    Args:
        base_i: Single-character base identity (A, C, G, U).
        base_j: Single-character base identity (A, C, G, U).
        family: Leontis-Westhof family abbreviation (default 'cWW').

    Returns:
        Weight in [0, 1].
    """
    w = LW_BP_WEIGHTS.get((base_i, base_j, family))
    if w is not None:
        return w
    w = LW_BP_WEIGHTS.get((base_j, base_i, family))
    if w is not None:
        return w
    return 0.4  # default non-canonical


# ---------------------------------------------------------------------------
# GNRA / UNCG Tetraloop Detection
# Source: Woese CR et al. (1990) PNAS 87:8467-8471
# ---------------------------------------------------------------------------
_GNRA_RE = re.compile(r'G[ACGU][AG]A')
_UNCG_RE = re.compile(r'U[ACGU]CG')
_UUCG_RE = re.compile(r'UUCG')


def detect_tetraloops(seq: str) -> dict[int, str]:
    """Return {start_position: loop_type} for all GNRA/UNCG tetraloops.

    Types: 'GNRA', 'UNCG', 'UUCG' (UUCG is thermodynamically most stable).
    Positions within a hairpin loop of length 4 only (enforced by caller).

    Args:
        seq: RNA sequence string (ACGU).

    Returns:
        Mapping from start position to tetraloop type name.
    """
    hits: dict[int, str] = {}
    for m in _UUCG_RE.finditer(seq):
        hits[m.start()] = 'UUCG'
    for m in _UNCG_RE.finditer(seq):
        if m.start() not in hits:
            hits[m.start()] = 'UNCG'
    for m in _GNRA_RE.finditer(seq):
        if m.start() not in hits:
            hits[m.start()] = 'GNRA'
    return hits


# ---------------------------------------------------------------------------
# Pseudoknot Cross-Pair Detection
# ---------------------------------------------------------------------------
def find_crossing_pairs(bp_list: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return base pairs that cross another pair (i.e. form pseudoknots).

    A pair (k, l) crosses (i, j) when i < k < j < l.

    Args:
        bp_list: List of (i, j) base pairs with i < j.

    Returns:
        Set of crossing (k, l) pairs.
    """
    crossing: set[tuple[int, int]] = set()
    sorted_bps = sorted(bp_list)
    for idx, (i, j) in enumerate(sorted_bps):
        for (k, l) in sorted_bps[idx + 1:]:
            if k >= j:
                break
            if i < k < j < l:
                crossing.add((k, l))
    return crossing


# ---------------------------------------------------------------------------
# Coaxial Stack Junction Detector
# ---------------------------------------------------------------------------
def detect_coaxial_junctions(
    helices: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return pairs of helix indices that are directly coaxially stackable.

    Two helices are coaxial if one ends exactly where the other begins
    (gap of 0 or 1 unpaired nucleotide at the junction).

    Args:
        helices: List of (start, end) helix spans.

    Returns:
        List of (helix_index_i, helix_index_j) pairs.
    """
    coaxial_pairs = []
    for i, (s1, e1) in enumerate(helices):
        for j, (s2, e2) in enumerate(helices):
            if i >= j:
                continue
            gap = min(abs(e1 - s2), abs(e2 - s1))
            if gap <= 1:
                coaxial_pairs.append((i, j))
    return coaxial_pairs


def _extract_helices(bp_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Extract helix spans from a list of base pairs.

    A helix is a run of consecutive base pairs (i, j), (i+1, j-1), ...

    Args:
        bp_list: List of (i, j) base pairs with i < j.

    Returns:
        List of (start, end) helix spans.
    """
    if not bp_list:
        return []
    bp_set = set(bp_list)
    sorted_bps = sorted(bp_list)
    visited: set[tuple[int, int]] = set()
    helices = []
    for (i, j) in sorted_bps:
        if (i, j) in visited:
            continue
        # Extend helix
        start_i, end_j = i, j
        cur_i, cur_j = i, j
        visited.add((cur_i, cur_j))
        while (cur_i + 1, cur_j - 1) in bp_set and cur_j - 1 > cur_i + 1:
            cur_i += 1
            cur_j -= 1
            visited.add((cur_i, cur_j))
        helices.append((start_i, cur_i))
    return helices


class ContactMapPredictor:
    """Predicts RNA contact maps using RG matrix field theory.

    For sequences of length L:
    1. Partition into blocks of size b = min(block_size, L//20)
    2. Solve saddle-point equations per block
    3. Reconstruct full contact map via hierarchical composition
    """

    def __init__(self, config: ContactMapConfig, bio_config: Optional[RNABiologyConstants] = None) -> None:
        self.config = config
        self._bio_config = bio_config or RNABiologyConstants()

    def predict(
        self, sequence: str, return_sparse: bool = True
    ) -> np.ndarray:
        r"""Predict contact probability map for an RNA sequence.

        Mathematical Basis:
            Block partition with size :math:`b = \min(300, L/20)`.
            Per-block saddle-point:
            :math:`M^* - t M^{*2} - u M^{*3} = 0`

            Reconstructed full contact probability:
            :math:`P_{ij} = P^{\text{block}}_{B(i),B(j)} \cdot
            P^{\text{local}}_{i|B(i)} \cdot P^{\text{local}}_{j|B(j)}`

        Args:
            sequence: RNA sequence string (ACGU).
            return_sparse: If True and L > 1000, return sparse matrix.

        Returns:
            Contact probability matrix, shape (L, L).
        """
        L = len(sequence)
        if L == 0:
            return np.empty((0, 0))

        encoded = encode_sequence(sequence)
        cfg = self.config

        # Determine block size
        b = min(cfg.block_size, max(L // 20, 10))
        b = max(b, 5)  # Minimum block size

        # Partition into blocks
        n_blocks = (L + b - 1) // b
        block_starts = [i * b for i in range(n_blocks)]
        block_ends = [min((i + 1) * b, L) for i in range(n_blocks)]

        logger.debug(
            "contact_map_blocks",
            L=L, b=b, n_blocks=n_blocks,
        )

        # Step 1: Compute local contact maps per block
        local_maps = []
        for bi in range(n_blocks):
            start, end = block_starts[bi], block_ends[bi]
            seq_block = encoded[start:end]
            block_size = end - start

            # Effective couplings with RG decay
            xi = cfg.correlation_length
            t_eff = cfg.t_coupling * np.exp(-block_size / xi)
            u_eff = cfg.u_coupling * np.exp(-2 * block_size / xi)

            P_local = rg_block_contact_map(
                seq_block, t_eff, u_eff, cfg.newton_max_iter, cfg.newton_tol
            )
            local_maps.append(P_local)

        # Step 2: Compute inter-block contact probabilities
        P_inter = self._compute_inter_block_contacts(
            encoded, n_blocks, block_starts, block_ends
        )

        # Step 3: Reconstruct full contact map
        P_full = np.zeros((L, L))

        # Fill local blocks
        for bi in range(n_blocks):
            start, end = block_starts[bi], block_ends[bi]
            bs = end - start
            P_full[start:end, start:end] = local_maps[bi][:bs, :bs]

        # Fill inter-block contacts (vectorised)
        for bi in range(n_blocks):
            for bj in range(bi + 1, n_blocks):
                si, ei = block_starts[bi], block_ends[bi]
                sj, ej = block_starts[bj], block_ends[bj]
                local_i_diag = np.diag(local_maps[bi])[: (ei - si)]
                local_j_diag = np.diag(local_maps[bj])[: (ej - sj)]
                block = P_inter[bi, bj] * np.outer(local_i_diag, local_j_diag)
                P_full[si:ei, sj:ej] = block
                P_full[sj:ej, si:ei] = block.T

        # Ensure symmetric and bounded
        P_full = (P_full + P_full.T) / 2.0
        np.clip(P_full, 0.0, 1.0, out=P_full)

        # --- Biology integration ---
        bio = self._bio_config

        # (1) Apply LW weight boost — vectorised via precomputed (4,4) matrix
        # Encoding: A=0, C=1, G=2, U=3
        _LW_MATRIX = np.array([
            [0.4,  0.35, 0.4, 0.9],   # A vs A,C,G,U
            [0.35, 0.4,  1.0, 0.4],   # C vs A,C,G,U
            [0.4,  1.0,  0.4, 0.7],   # G vs A,C,G,U
            [0.9,  0.4,  0.7, 0.4],   # U vs A,C,G,U
        ])
        lw_full = _LW_MATRIX[encoded[:, None], encoded[None, :]]  # (L, L)
        upper_triangle_mask = np.arange(L)[:, None] < np.arange(L)[None, :]
        sig_mask = (P_full > 0.1) & upper_triangle_mask
        P_full[sig_mask] *= lw_full[sig_mask]
        # Mirror to lower triangle
        P_full.T[sig_mask] *= lw_full[sig_mask]

        # (2) GNRA / tetraloop bonus
        tetraloop_pos = detect_tetraloops(sequence)
        for pos, loop_type in tetraloop_pos.items():
            if loop_type in ('GNRA', 'UUCG'):
                window = slice(max(0, pos - 15), min(L, pos + 20))
                P_full[pos:min(pos + 4, L), window] *= (1.0 + bio.weight_gnra_bonus)

        # (3) Crossing pairs → pseudoknot weight
        bp_i, bp_j = np.where(sig_mask)
        bp_list = [(int(i), int(j)) for i, j in zip(bp_i, bp_j)]
        crossing = find_crossing_pairs(bp_list)
        for (ci, cj) in crossing:
            P_full[ci, cj] *= bio.weight_pseudoknot
            P_full[cj, ci] *= bio.weight_pseudoknot

        # (4) Coaxial stacking bonus
        helices = _extract_helices(bp_list)
        coaxial_pairs = detect_coaxial_junctions(helices)
        for (hi, hj) in coaxial_pairs:
            si, ei = helices[hi]
            sj, ej = helices[hj]
            # Boost contacts between the junction ends of the two helices
            junction_window_i = slice(max(0, ei - 2), min(L, ei + 2))
            junction_window_j = slice(max(0, sj - 2), min(L, sj + 2))
            P_full[junction_window_i, junction_window_j] *= (1.0 + bio.weight_coaxial_bonus)
            P_full[junction_window_j, junction_window_i] *= (1.0 + bio.weight_coaxial_bonus)

        # (5) A-minor bonus: adenosines contacting helices at long range
        helix_positions = set()
        for (hs, he) in helices:
            for pos in range(hs, he + 1):
                if pos < L:
                    helix_positions.add(pos)
        for idx in range(L):
            if sequence[idx] == 'A':
                for hp in helix_positions:
                    if abs(idx - hp) > 4 and P_full[idx, hp] > 0.1:
                        P_full[idx, hp] *= (1.0 + bio.weight_a_minor_bonus)
                        P_full[hp, idx] *= (1.0 + bio.weight_a_minor_bonus)

        # Re-symmetrize and clip after biology adjustments
        P_full = (P_full + P_full.T) / 2.0
        np.clip(P_full, 0.0, 1.0, out=P_full)

        # Sparsify for large sequences
        if return_sparse and L > 1000:
            P_full[P_full < 0.01] = 0.0
            return sparse.csr_matrix(P_full)

        return P_full

    def _compute_inter_block_contacts(
        self,
        encoded: np.ndarray,
        n_blocks: int,
        block_starts: list,
        block_ends: list,
    ) -> np.ndarray:
        """Compute inter-block contact probabilities.

        Uses sequence composition similarity and distance decay.

        Returns:
            Inter-block probability matrix, shape (n_blocks, n_blocks).
        """
        P_inter = np.zeros((n_blocks, n_blocks))

        for bi in range(n_blocks):
            for bj in range(bi + 1, n_blocks):
                si, ei = block_starts[bi], block_ends[bi]
                sj, ej = block_starts[bj], block_ends[bj]

                # Sequence separation decay
                mid_i = (si + ei) / 2.0
                mid_j = (sj + ej) / 2.0
                sep = abs(mid_j - mid_i)
                xi = self.config.correlation_length

                # Exponential decay with sequence separation
                p = np.exp(-sep / xi) * 0.5

                # Base composition complementarity boost
                comp_i = np.bincount(encoded[si:ei], minlength=4) / max(ei - si, 1)
                comp_j = np.bincount(encoded[sj:ej], minlength=4) / max(ej - sj, 1)
                # A-U and G-C complementarity
                complement = comp_i[0] * comp_j[3] + comp_i[3] * comp_j[0] + \
                             comp_i[1] * comp_j[2] + comp_i[2] * comp_j[1]
                p *= (1.0 + complement)

                P_inter[bi, bj] = min(p, 1.0)
                P_inter[bj, bi] = P_inter[bi, bj]

        return P_inter

    def extract_genus(self, contact_map: np.ndarray, sequence: str = "") -> int:
        r"""Extract topological genus from contact map via Gauss code rank.

        Mathematical Basis:
            :math:`g = \max(0, \lfloor(|B| + 1 -
            \text{rank}_{\mathbb{F}_2}(G)) / 2\rfloor)`

            where :math:`G` is the arc crossing matrix over :math:`\mathbb{F}_2`.

        Args:
            contact_map: Contact probability matrix, shape (L, L).
            sequence: RNA sequence string (unused, kept for API compatibility).

        Returns:
            Estimated genus (int >= 0).
        """
        if sparse.issparse(contact_map):
            cm = contact_map.toarray()
        else:
            cm = contact_map
        L = cm.shape[0]
        if L < 5:
            return 0

        threshold = 0.3
        bp_i, bp_j = np.where(
            (cm > threshold)
            & (np.arange(L)[:, None] < np.arange(L)[None, :])
        )
        valid = (bp_j - bp_i) >= 4
        arc_i = bp_i[valid].astype(np.int64)
        arc_j = bp_j[valid].astype(np.int64)
        if len(arc_i) == 0:
            return 0
        return int(compute_genus_gauss_code(arc_i, arc_j, L))
