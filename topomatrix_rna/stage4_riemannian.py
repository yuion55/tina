"""Stage 4: Riemannian Torsion Refinement.

Refines RNA 3D structures via Riemannian ADAM optimizer on the torus
manifold T^7 (7 torsion angles per residue), with Störmer-Verlet
symplectic integration and Hogwild parallel updates.

References:
    [14] Bonnabel S (2013). IEEE Trans Autom Control 58:2217-2229 — Riemannian SGD
    [15] Becigneul G, Ganea OE (2019). ICLR 2019 — Riemannian ADAM
    [16] Leimkuhler B, Reich S (2004). Cambridge Univ Press — symplectic integrators
    [17] Recht B et al. (2011). NIPS 24 — Hogwild! parallel SGD
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import structlog

from .config import RiemannianConfig
from .numba_kernels import (
    exp_map_torus,
    parallel_transport_torus,
    rsrnasp1_energy_block,
    rsrnasp1_gradient,
    stoermer_verlet_step,
    torus_geodesic_distance,
)

logger = structlog.get_logger(__name__)


class RiemannianRefiner:
    r"""Refines RNA torsion angles on the Riemannian manifold :math:`(\mathbb{T}^7)^L`.

    Uses Riemannian ADAM with parallel transport for momentum,
    and optional Störmer-Verlet symplectic integration.
    """

    def __init__(self, config: RiemannianConfig) -> None:
        self.config = config

    def refine(
        self,
        theta_init: np.ndarray,
        seq_encoded: np.ndarray,
        energy_fn: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""Refine torsion angles using Riemannian ADAM on :math:`\mathbb{T}^7`.

        Mathematical Basis:
            :math:`m_t = \beta_1 \cdot \text{PT}(m_{t-1}) + (1-\beta_1) \cdot
            \nabla_{\mathcal{M}} E(\theta_t)`

            :math:`v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot
            \|\nabla_{\mathcal{M}} E(\theta_t)\|^2`

            :math:`\theta_{t+1} = \text{Exp}_{\theta_t}
            (-\frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t)`

        Args:
            theta_init: Initial torsion angles, shape (L, 7).
            seq_encoded: Encoded sequence (0-3), shape (L,).
            energy_fn: Optional custom energy function. Uses rsRNASP1 if None.

        Returns:
            Tuple of (optimized_theta, final_energy).
        """
        cfg = self.config
        L = theta_init.shape[0]
        n_torsions = theta_init.shape[1]

        theta = theta_init.copy()
        m = np.zeros_like(theta)  # First moment
        v = np.zeros((L,))  # Second moment (scalar per residue)

        # Keep a reference to the caller-supplied function (None → analytical grad)
        _custom_energy_fn = energy_fn
        if energy_fn is None:
            energy_fn = lambda t: float(rsrnasp1_energy_block(t, seq_encoded))

        best_theta = theta.copy()
        best_energy = energy_fn(theta)

        logger.info(
            "riemannian_refine_start",
            L=L, n_steps=cfg.n_steps, initial_energy=best_energy,
        )

        for step in range(cfg.n_steps):
            # Compute gradient: analytical O(L) when no custom energy_fn,
            # finite differences O(L·7) otherwise.
            grad = self._compute_gradient(theta, seq_encoded, _custom_energy_fn)

            # Riemannian ADAM update (block-coordinate for Hogwild)
            block_start = (step * cfg.block_size) % L
            block_end = min(block_start + cfg.block_size, L)

            for i in range(block_start, block_end):
                # Parallel transport momentum (identity on flat torus)
                m_transported = m[i].copy()

                # Update first moment
                m[i] = cfg.beta1 * m_transported + (1 - cfg.beta1) * grad[i]

                # Update second moment
                grad_norm_sq = float(np.sum(grad[i] ** 2))
                v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * grad_norm_sq

                # Bias correction
                m_hat = m[i] / (1 - cfg.beta1 ** (step + 1))
                v_hat = v[i] / (1 - cfg.beta2 ** (step + 1))

                # Riemannian update via exponential map
                update = -cfg.learning_rate / (np.sqrt(v_hat) + cfg.epsilon) * m_hat
                theta[i] = exp_map_torus(theta[i], update)

            # Track best
            current_energy = energy_fn(theta)
            if current_energy < best_energy:
                best_energy = current_energy
                best_theta = theta.copy()

            if (step + 1) % 100 == 0:
                logger.debug(
                    "riemannian_step",
                    step=step + 1, energy=current_energy, best_energy=best_energy,
                )

        logger.info(
            "riemannian_refine_done",
            final_energy=best_energy,
        )

        return best_theta, best_energy

    def refine_symplectic(
        self,
        theta_init: np.ndarray,
        seq_encoded: np.ndarray,
        energy_fn: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""Refine using Störmer-Verlet symplectic integration.

        Mathematical Basis:
            :math:`p^{n+1/2} = p^n - \frac{h}{2}\nabla_\theta E(\theta^n)`
            :math:`\theta^{n+1} = (\theta^n + h \cdot p^{n+1/2}) \bmod 2\pi`
            :math:`p^{n+1} = p^{n+1/2} - \frac{h}{2}\nabla_\theta E(\theta^{n+1})`

        The gradient is recomputed at the new position :math:`\theta^{n+1}` for
        the second half-step, ensuring true symplectic integration.

        Args:
            theta_init: Initial torsion angles, shape (L, 7).
            seq_encoded: Encoded sequence, shape (L,).
            energy_fn: Optional custom energy function.

        Returns:
            Tuple of (optimized_theta, final_energy).
        """
        cfg = self.config
        theta = theta_init.copy()
        p = np.zeros_like(theta)

        # Keep reference to caller-supplied function (None → analytical grad)
        _custom_energy_fn = energy_fn
        if energy_fn is None:
            energy_fn = lambda t: float(rsrnasp1_energy_block(t, seq_encoded))

        best_theta = theta.copy()
        best_energy = energy_fn(theta)

        for step in range(cfg.n_steps):
            grad_old = self._compute_gradient(theta, seq_encoded, _custom_energy_fn)
            p -= 0.5 * cfg.symplectic_h * grad_old
            theta = (theta + cfg.symplectic_h * p) % (2.0 * np.pi)
            grad_new = self._compute_gradient(theta, seq_encoded, _custom_energy_fn)
            p -= 0.5 * cfg.symplectic_h * grad_new

            current_energy = energy_fn(theta)
            if current_energy < best_energy:
                best_energy = current_energy
                best_theta = theta.copy()

        return best_theta, best_energy

    def _compute_gradient(
        self,
        theta: np.ndarray,
        seq_encoded: np.ndarray,
        energy_fn: Optional[Callable] = None,
        delta: float = 1e-4,
    ) -> np.ndarray:
        """Compute gradient of the rsRNASP1 energy w.r.t. torsion angles.

        Uses the analytical ``rsrnasp1_gradient`` kernel (O(L)) when no custom
        energy function is provided, falling back to central finite differences
        (O(L·7)) only for custom energy functions.

        Args:
            theta: Current torsion angles, shape (L, 7).
            seq_encoded: Encoded sequence, shape (L,).
            energy_fn: Optional custom energy function. If ``None``, the
                analytical gradient is used.
            delta: Finite difference step size (used only when energy_fn is
                not None).

        Returns:
            Gradient array, shape (L, 7).
        """
        if energy_fn is None:
            return rsrnasp1_gradient(theta, seq_encoded)

        # Finite differences only for custom energy functions
        L, n_tor = theta.shape
        grad = np.zeros_like(theta)
        for i in range(L):
            for k in range(n_tor):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[i, k] += delta
                theta_minus[i, k] -= delta
                grad[i, k] = (energy_fn(theta_plus) - energy_fn(theta_minus)) / (2 * delta)
        return grad

    def torsion_to_coords(
        self, theta: np.ndarray, template_coords: np.ndarray
    ) -> np.ndarray:
        """Convert torsion angles to C3' Cartesian coordinates via NeRF.

        Uses the Natural Extension Reference Frame (NeRF) algorithm to
        build a chain of C3' atoms that respects the given backbone torsion
        angles while maintaining fixed bond lengths and bond angles.

        Args:
            theta: Torsion angles, shape (L, 7).
            template_coords: Template C3' coordinates used to seed the first
                three residues, shape (M, 3).

        Returns:
            Predicted C3' coordinates, shape (L, 3).
        """
        L = theta.shape[0]
        coords = np.zeros((L, 3))

        # Seed with template (up to 3 residues)
        if template_coords.shape[0] >= 3:
            coords[:3] = template_coords[:3].copy()
        else:
            coords[0] = np.array([0.0, 0.0, 0.0])
            coords[1] = np.array([5.9, 0.0, 0.0])
            coords[2] = np.array([11.8, 0.0, 0.0])

        BOND_LENGTH = 5.9      # Å, C3'–C3' virtual bond
        BOND_ANGLE = 1.745     # rad, ~100°

        COMPLEMENT_ANGLE = np.pi - BOND_ANGLE  # ~0.204 rad (complement of ~100°)

        for i in range(3, L):
            a, b, c = coords[i - 3], coords[i - 2], coords[i - 1]
            torsion = theta[i, 0]  # Backbone torsion (alpha)

            bc = c - b
            bc_norm = np.linalg.norm(bc)
            bc = bc / (bc_norm + 1e-10)

            n_vec = np.cross(b - a, bc)
            n_norm = np.linalg.norm(n_vec)
            n_vec = n_vec / n_norm if n_norm > 1e-10 else np.array([0.0, 0.0, 1.0])

            d = np.array([
                -BOND_LENGTH * np.cos(COMPLEMENT_ANGLE),
                 BOND_LENGTH * np.cos(torsion) * np.sin(COMPLEMENT_ANGLE),
                 BOND_LENGTH * np.sin(torsion) * np.sin(COMPLEMENT_ANGLE),
            ])
            M = np.column_stack([bc, np.cross(n_vec, bc), n_vec])
            coords[i] = c + M @ d

        return coords
