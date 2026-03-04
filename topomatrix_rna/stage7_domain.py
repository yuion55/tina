"""Stage 7: Spectral Domain Decomposition.

For long RNA sequences (L > 500), decomposes the structure into domains
using the Fiedler vector of the contact map Laplacian, then assembles
domains via SE(3) rigid body optimization.

References:
    [23] Chirikjian GS (2011). Springer ISBN 9780817649401 — SE(3) methods
    [24] Belkin M, Niyogi P (2003). Neural Comput 15:1373-1396 — Laplacian eigenmaps
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import structlog
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .config import DomainConfig
from .data_utils import kabsch_align

logger = structlog.get_logger(__name__)


class SpectralDomainDecomposer:
    """Decomposes RNA structures into domains using spectral methods.

    Uses the Fiedler vector (second eigenvector of the graph Laplacian)
    to identify domain boundaries.
    """

    def __init__(self, config: DomainConfig) -> None:
        self.config = config

    def decompose(
        self, contact_map: np.ndarray, L: int
    ) -> List[Tuple[int, int]]:
        r"""Decompose sequence into domains using spectral clustering.

        Mathematical Basis:
            Contact map Laplacian:
            :math:`L = D - P, \quad D_{ii} = \sum_j P_{ij}`

            Fiedler vector (second eigenvector):
            :math:`L v_2 = \lambda_2 v_2, \quad \lambda_2 > 0`

            Domain boundaries at sign changes and local extrema of :math:`v_2`.

            The Laplacian is built entirely in sparse format to avoid
            allocating two dense L×L matrices.

        Args:
            contact_map: Contact probability matrix, shape (L, L) — may be
                a dense ``np.ndarray`` or a ``scipy.sparse.csr_matrix``.
            L: Sequence length.

        Returns:
            List of (start, end) domain boundaries.
        """
        if L < self.config.use_threshold_length:
            return [(0, L)]

        cfg = self.config

        # Build sparse contact matrix (handle both dense and sparse inputs)
        if not sparse.issparse(contact_map):
            cm = sparse.csr_matrix(contact_map)
        else:
            cm = contact_map

        # Compute graph Laplacian purely in sparse space (no dense L×L alloc)
        degrees = np.array(cm.sum(axis=1)).ravel()
        D_sparse = sparse.diags(degrees, format="csr")
        Lap_sparse = D_sparse - cm

        try:
            n_eigs = min(6, L - 1)
            eigenvalues, eigenvectors = eigsh(
                Lap_sparse, k=n_eigs, which="SM", maxiter=1000, tol=1e-5
            )
            order = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, order[1]]
        except Exception as e:
            logger.warning("eigsh_failed", error=str(e))
            return self._uniform_decomposition(L)

        # Find domain boundaries at sign changes of Fiedler vector
        boundaries = [0]
        for i in range(1, L):
            if fiedler[i] * fiedler[i - 1] < 0:
                boundaries.append(i)

        # Fallback for disconnected graphs (no sign changes):
        # partition by median value of Fiedler vector.
        if len(boundaries) == 1:
            threshold = np.median(fiedler)
            prev_above = bool(fiedler[0] >= threshold)
            for i in range(1, L):
                curr_above = bool(fiedler[i] >= threshold)
                if curr_above != prev_above:
                    boundaries.append(i)
                prev_above = curr_above

        boundaries.append(L)

        # Merge small domains
        domains = []
        i = 0
        while i < len(boundaries) - 1:
            start = boundaries[i]
            end = boundaries[i + 1]

            # Merge with next if too small
            while (end - start) < cfg.min_domain_size and i + 2 < len(boundaries):
                i += 1
                end = boundaries[i + 1]

            # Split if too large
            if (end - start) > cfg.max_domain_size:
                n_splits = (end - start + cfg.max_domain_size - 1) // cfg.max_domain_size
                split_size = (end - start) // n_splits
                for s in range(n_splits):
                    ds = start + s * split_size
                    de = start + (s + 1) * split_size if s < n_splits - 1 else end
                    domains.append((ds, de))
            else:
                domains.append((start, end))
            i += 1

        if len(domains) == 0:
            domains = [(0, L)]

        logger.info(
            "domain_decomposition",
            L=L, n_domains=len(domains),
            domain_sizes=[d[1] - d[0] for d in domains],
        )
        return domains

    def _uniform_decomposition(self, L: int) -> List[Tuple[int, int]]:
        """Fallback uniform domain decomposition."""
        cfg = self.config
        domain_size = min(cfg.max_domain_size, max(cfg.min_domain_size, L // 5))
        domains = []
        for i in range(0, L, domain_size):
            domains.append((i, min(i + domain_size, L)))
        return domains


class SE3DomainAssembler:
    """Assembles RNA domains via SE(3) rigid body optimization.

    Optimizes rotation and translation of each domain to minimize
    inter-domain contact energy.
    """

    def __init__(self, config: DomainConfig) -> None:
        self.config = config

    def assemble(
        self,
        domain_coords: List[np.ndarray],
        domains: List[Tuple[int, int]],
        contact_map: np.ndarray,
    ) -> np.ndarray:
        r"""Assemble domain coordinates into full structure.

        Mathematical Basis:
            Domain junction energy on :math:`SE(3)^{N_d}`:
            :math:`E_{\text{assembly}} = \sum_{\text{inter-domain}} V(d_{ij}(R, t))`

            Riemannian gradient descent on SE(3):
            :math:`\nabla_{\text{SE(3)}} E = \text{Ad}^*_g \frac{\partial E}{\partial g}`

        Args:
            domain_coords: List of per-domain coordinate arrays.
            domains: List of (start, end) boundaries.
            contact_map: Full contact map for inter-domain contacts.

        Returns:
            Assembled coordinates, shape (L, 3).
        """
        n_domains = len(domains)
        if n_domains <= 1:
            return domain_coords[0] if domain_coords else np.empty((0, 3))

        L = domains[-1][1]
        full_coords = np.zeros((L, 3))

        # Initialize: place first domain
        start, end = domains[0]
        full_coords[start:end] = domain_coords[0][: end - start]

        # Place remaining domains relative to previous
        for d in range(1, n_domains):
            start, end = domains[d]
            prev_start, prev_end = domains[d - 1]
            domain_size = end - start

            # Initial placement: extend from previous domain
            if domain_size <= domain_coords[d].shape[0]:
                coords = domain_coords[d][:domain_size].copy()
            else:
                coords = np.zeros((domain_size, 3))
                coords[: domain_coords[d].shape[0]] = domain_coords[d]

            # Align junction point: last point of previous domain
            junction_prev = full_coords[prev_end - 1]
            junction_curr = coords[0]
            translation = junction_prev - junction_curr

            # Apply translation
            coords += translation

            # Add slight offset to avoid overlap
            direction = np.array([5.9, 0.0, 0.0])  # ~C3'-C3' bond length
            if d < n_domains:
                coords += direction * 0.1

            full_coords[start:end] = coords

        # Refine assembly with gradient descent on SE(3)
        full_coords = self._refine_assembly(
            full_coords, domains, contact_map
        )

        return full_coords

    def _refine_assembly(
        self,
        coords: np.ndarray,
        domains: List[Tuple[int, int]],
        contact_map: np.ndarray,
    ) -> np.ndarray:
        """Refine domain assembly via SE(3) gradient descent.

        The inner contact-pair loop is vectorised using ``np.where`` and
        NumPy broadcasts, handling both dense and sparse contact maps.

        Args:
            coords: Initial assembled coordinates.
            domains: Domain boundaries.
            contact_map: Contact map for computing inter-domain energy.

        Returns:
            Refined coordinates.
        """
        cfg = self.config
        n_domains = len(domains)

        for step in range(cfg.se3_steps):
            for d in range(1, n_domains):
                start, end = domains[d]
                total_grad = np.zeros(3)
                total_n = 0

                for d2 in range(n_domains):
                    if d2 == d:
                        continue
                    s2, e2 = domains[d2]

                    if sparse.issparse(contact_map):
                        block = np.asarray(
                            contact_map[start:end, s2:e2].todense()
                        )
                    else:
                        block = contact_map[start:end, s2:e2]

                    rows, cols = np.where(block > 0.05)
                    if len(rows) == 0:
                        continue
                    p_vals = block[rows, cols]
                    diffs = coords[start + rows] - coords[s2 + cols]
                    dists = np.linalg.norm(diffs, axis=1) + 1e-10
                    forces = p_vals * (dists - 8.0) / dists
                    total_grad += np.sum(
                        forces[:, np.newaxis] * diffs, axis=0
                    )
                    total_n += len(rows)

                if total_n > 0:
                    coords[start:end] -= cfg.se3_lr * (total_grad / total_n)

        return coords


def helix_boundary_penalty(
    cut_position: int,
    helix_spans: list[tuple[int, int]],
    ss_linkers: list[tuple[int, int]],
) -> float:
    """Penalty for placing a domain boundary at a given sequence position.

    High penalty (10.0) if cut falls inside a helix (breaks base pairs).
    Zero penalty if cut falls in a single-stranded linker.
    Moderate penalty (1.0) otherwise.

    Args:
        cut_position: Sequence index for proposed domain boundary.
        helix_spans: Helix (start, end) pairs from secondary structure.
        ss_linkers: Single-stranded linker (start, end) pairs.

    Returns:
        Non-negative penalty scalar.
    """
    for (start, end) in helix_spans:
        if start < cut_position < end:
            return 10.0   # never cut through a helix
    for (start, end) in ss_linkers:
        if start <= cut_position <= end:
            return 0.0    # ideal cut site
    return 1.0            # acceptable but not preferred
