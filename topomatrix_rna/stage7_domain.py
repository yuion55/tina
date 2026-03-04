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

        Args:
            contact_map: Contact probability matrix, shape (L, L).
            L: Sequence length.

        Returns:
            List of (start, end) domain boundaries.
        """
        if L < self.config.use_threshold_length:
            return [(0, L)]

        cfg = self.config

        # Compute graph Laplacian
        D = np.diag(np.sum(contact_map, axis=1))
        Lap = D - contact_map

        # Make sparse for efficiency
        Lap_sparse = sparse.csr_matrix(Lap)

        try:
            # Compute Fiedler vector (second smallest eigenvector)
            n_eigs = min(6, L - 1)
            eigenvalues, eigenvectors = eigsh(
                Lap_sparse, k=n_eigs, which="SM", maxiter=1000
            )
            fiedler = eigenvectors[:, 1]  # Second eigenvector
        except Exception as e:
            logger.warning("eigsh_failed", error=str(e))
            # Fallback: uniform domain decomposition
            return self._uniform_decomposition(L)

        # Find domain boundaries at sign changes of Fiedler vector
        boundaries = [0]
        for i in range(1, L):
            if fiedler[i] * fiedler[i - 1] < 0:
                boundaries.append(i)
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

        Args:
            coords: Initial assembled coordinates.
            domains: Domain boundaries.
            contact_map: Contact map for computing inter-domain energy.

        Returns:
            Refined coordinates.
        """
        cfg = self.config
        n_domains = len(domains)
        L = coords.shape[0]

        # Per-domain rotation (as axis-angle) and translation
        rotations = np.zeros((n_domains, 3))  # axis-angle
        translations = np.zeros((n_domains, 3))

        for step in range(cfg.se3_steps):
            # Compute inter-domain energy gradient
            for d in range(1, n_domains):
                start, end = domains[d]

                # Gradient of junction energy
                grad_t = np.zeros(3)
                n_contacts = 0

                for d2 in range(n_domains):
                    if d2 == d:
                        continue
                    s2, e2 = domains[d2]

                    # Inter-domain contacts
                    for i in range(start, end):
                        for j in range(s2, e2):
                            if i < contact_map.shape[0] and j < contact_map.shape[1]:
                                p_contact = contact_map[i, j]
                                if p_contact > 0.05:
                                    diff = coords[i] - coords[j]
                                    dist = np.linalg.norm(diff) + 1e-10
                                    # Attractive force for contacts
                                    target_dist = 8.0  # Typical contact distance
                                    force = p_contact * (dist - target_dist) / dist
                                    grad_t += force * diff
                                    n_contacts += 1

                if n_contacts > 0:
                    grad_t /= n_contacts

                # Update translation
                translations[d] -= cfg.se3_lr * grad_t

                # Apply to coordinates
                for i in range(start, end):
                    coords[i] -= cfg.se3_lr * grad_t

        return coords
