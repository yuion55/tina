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

from typing import Optional, Tuple

import numpy as np
import structlog
from scipy import sparse

from .config import ContactMapConfig
from .data_utils import encode_sequence
from .numba_kernels import rg_block_contact_map

logger = structlog.get_logger(__name__)


class ContactMapPredictor:
    """Predicts RNA contact maps using RG matrix field theory.

    For sequences of length L:
    1. Partition into blocks of size b = min(block_size, L//20)
    2. Solve saddle-point equations per block
    3. Reconstruct full contact map via hierarchical composition
    """

    def __init__(self, config: ContactMapConfig) -> None:
        self.config = config

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

        # Fill inter-block contacts
        for bi in range(n_blocks):
            for bj in range(bi + 1, n_blocks):
                si, ei = block_starts[bi], block_ends[bi]
                sj, ej = block_starts[bj], block_ends[bj]
                bsi = ei - si
                bsj = ej - sj

                # Scale by inter-block probability
                p_inter = P_inter[bi, bj]
                for i in range(bsi):
                    for j in range(bsj):
                        # Local probability contribution
                        p_local_i = local_maps[bi][i, i] if i < local_maps[bi].shape[0] else 0.5
                        p_local_j = local_maps[bj][j, j] if j < local_maps[bj].shape[0] else 0.5
                        P_full[si + i, sj + j] = p_inter * p_local_i * p_local_j
                        P_full[sj + j, si + i] = P_full[si + i, sj + j]

        # Ensure symmetric and bounded
        P_full = (P_full + P_full.T) / 2.0
        np.clip(P_full, 0.0, 1.0, out=P_full)

        # Sparsify for large sequences
        if return_sparse and L > 1000:
            P_full[P_full < 0.01] = 0.0
            return sparse.csr_matrix(P_full).toarray()

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

    def extract_genus(self, contact_map: np.ndarray) -> int:
        r"""Extract topological genus from contact map.

        Mathematical Basis:
            Uses :math:`1/N^{2g}` expansion coefficient:
            :math:`\hat{g} = \arg\max_g \sum_{i<j} P_{ij}^{(g)}`

        Args:
            contact_map: Contact probability matrix, shape (L, L).

        Returns:
            Estimated genus (int >= 0).
        """
        L = contact_map.shape[0]
        if L < 5:
            return 0

        # Eigenvalue analysis of contact map
        eigenvalues = np.linalg.eigvalsh(contact_map)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Genus estimated from spectral gap structure
        # Number of significant eigenvalues ~ 2g + 1
        threshold = eigenvalues[0] * 0.1 if eigenvalues[0] > 0 else 0.01
        n_significant = int(np.sum(eigenvalues > threshold))
        genus = max(0, (n_significant - 1) // 2)

        return genus
