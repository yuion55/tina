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

from .config import RiemannianConfig, RNABiologyConstants
from .numba_kernels import (
    exp_map_torus,
    parallel_transport_torus,
    rsrnasp1_energy_block,
    rsrnasp1_gradient,
    stoermer_verlet_step,
    torus_geodesic_distance,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# RNA backbone suite conformer cluster means (degrees)
# Format: suite_name -> {'delta_prev', 'epsilon', 'zeta', 'alpha', 'beta', 'gamma', 'delta'}
# Source: Richardson JS et al. (2008) RNA 14:465-481
# ---------------------------------------------------------------------------
SUITE_CLUSTER_MEANS: dict[str, dict[str, float]] = {
    '1a': {'delta_prev': 83, 'epsilon': 212, 'zeta': 289, 'alpha': -68, 'beta': 178, 'gamma': 55,  'delta': 83},   # A-form
    '1L': {'delta_prev': 83, 'epsilon': 245, 'zeta': 179, 'alpha': -68, 'beta': 163, 'gamma': 50,  'delta': 84},
    '1m': {'delta_prev': 86, 'epsilon': 218, 'zeta': 291, 'alpha': -70, 'beta': 168, 'gamma': 51,  'delta': 145},
    '5z': {'delta_prev': 83, 'epsilon': 192, 'zeta': 264, 'alpha': -58, 'beta': 172, 'gamma': 51,  'delta': 86},   # S-motif
    '4s': {'delta_prev': 145, 'epsilon': 264, 'zeta': -70, 'alpha': -68, 'beta': 173, 'gamma': 175, 'delta': 83},
    '#a': {'delta_prev': 145, 'epsilon': 210, 'zeta': 296, 'alpha': -66, 'beta': 177, 'gamma': 54,  'delta': 83},
    '7r': {'delta_prev': 83, 'epsilon': 304, 'zeta': 67,  'alpha': -65, 'beta': 175, 'gamma': 55,  'delta': 83},
    '6n': {'delta_prev': 83, 'epsilon': 268, 'zeta': 288, 'alpha': -66, 'beta': 173, 'gamma': 60,  'delta': 145},
    '0a': {'delta_prev': 83, 'epsilon': 223, 'zeta': 180, 'alpha': 51,  'beta': 173, 'gamma': 51,  'delta': 83},
    '&a': {'delta_prev': 83, 'epsilon': 210, 'zeta': 300, 'alpha': -67, 'beta': 160, 'gamma': 54,  'delta': 83},
}

SUITE_ANGLE_KEYS = ['delta_prev', 'epsilon', 'zeta', 'alpha', 'beta', 'gamma', 'delta']
SUITE_TOLERANCE_DEG = 40.0   # degrees; Mahalanobis distance threshold for penalty onset


class RiemannianRefiner:
    r"""Refines RNA torsion angles on the Riemannian manifold :math:`(\mathbb{T}^7)^L`.

    Uses Riemannian ADAM with parallel transport for momentum,
    and optional Störmer-Verlet symplectic integration.
    """

    def __init__(self, config: RiemannianConfig, bio_config: Optional[RNABiologyConstants] = None) -> None:
        self.config = config
        self._bio_config = bio_config or RNABiologyConstants()

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
                # Backbone biology penalties — scale gradient BEFORE ADAM
                # consumes it, so the penalty actually influences the update
                bio = self._bio_config
                delta_deg = np.degrees(theta[i, 3]) if theta.shape[1] > 3 else 0.0
                chi_deg = np.degrees(theta[i, 6]) if theta.shape[1] > 6 else 0.0
                pp = self.sugar_pucker_penalty(delta_deg, config=bio)
                cp = self.chi_syn_penalty(chi_deg)
                angle_dict = {
                    k: np.degrees(theta[i, n])
                    for n, k in enumerate(SUITE_ANGLE_KEYS) if n < theta.shape[1]
                }
                sp = self.suite_conformer_penalty(angle_dict, config=bio)
                total_bio_pen = pp + cp + sp
                grad[i] *= (1.0 + total_bio_pen * 0.01)

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
            grad_old = self._apply_bio_penalty_to_grad(grad_old, theta)
            p -= 0.5 * cfg.symplectic_h * grad_old
            theta = (theta + cfg.symplectic_h * p) % (2.0 * np.pi)
            grad_new = self._compute_gradient(theta, seq_encoded, _custom_energy_fn)
            grad_new = self._apply_bio_penalty_to_grad(grad_new, theta)
            p -= 0.5 * cfg.symplectic_h * grad_new

            current_energy = energy_fn(theta)
            if current_energy < best_energy:
                best_energy = current_energy
                best_theta = theta.copy()

        return best_theta, best_energy

    def _apply_bio_penalty_to_grad(
        self, grad: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Scale gradient by backbone biology penalties for each residue.

        Args:
            grad: Gradient array, shape (L, n_torsions).
            theta: Current torsion angles, shape (L, n_torsions).

        Returns:
            Penalty-scaled gradient, shape (L, n_torsions).
        """
        bio = self._bio_config
        grad = grad.copy()
        for i in range(grad.shape[0]):
            delta_deg = np.degrees(theta[i, 3]) if theta.shape[1] > 3 else 0.0
            chi_deg = np.degrees(theta[i, 6]) if theta.shape[1] > 6 else 0.0
            pp = self.sugar_pucker_penalty(delta_deg, config=bio)
            cp = self.chi_syn_penalty(chi_deg)
            angle_dict = {
                k: np.degrees(theta[i, n])
                for n, k in enumerate(SUITE_ANGLE_KEYS) if n < theta.shape[1]
            }
            sp = self.suite_conformer_penalty(angle_dict, config=bio)
            total_bio_pen = pp + cp + sp
            grad[i] *= (1.0 + total_bio_pen * 0.01)
        return grad

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

    def suite_conformer_penalty(
        self, angles_deg: dict[str, float], config: RNABiologyConstants | None = None
    ) -> float:
        """Penalise backbone angles that fall outside all known suite conformer clusters.

        Computes L1 distance from each suite cluster mean and returns the
        excess beyond SUITE_TOLERANCE_DEG for the nearest cluster.
        Returns 0.0 if the angles are within any known suite cluster.

        Source: Richardson JS et al. (2008) RNA 14:465-481.

        Args:
            angles_deg: Dict with keys matching SUITE_ANGLE_KEYS, values in degrees.
            config: Optional RNABiologyConstants; uses default if None.

        Returns:
            Non-negative penalty value (kcal/mol units, scaled by weight_suite_penalty).
        """
        cfg = config or self._bio_config
        best_dist = float('inf')
        for suite_name, means in SUITE_CLUSTER_MEANS.items():
            dist = 0.0
            for key in SUITE_ANGLE_KEYS:
                if key in angles_deg and key in means:
                    diff = abs(angles_deg[key] - means[key])
                    diff = min(diff, 360.0 - diff)   # handle periodicity
                    dist += diff
            if dist < best_dist:
                best_dist = dist
        excess = max(0.0, best_dist - SUITE_TOLERANCE_DEG)
        return excess * cfg.weight_suite_penalty

    def sugar_pucker_penalty(
        self, delta_deg: float, position_type: str = 'helix',
        config: RNABiologyConstants | None = None,
    ) -> float:
        """Penalise δ torsion angles outside valid C3'-endo / C2'-endo windows.

        Source: Altona C, Sundaralingam M (1972) JACS 94:8205-8212.

        Args:
            delta_deg: δ torsion angle in degrees.
            position_type: 'helix' (default, C3'-endo expected) or
                           'non_canonical' (C2'-endo allowed).
            config: Optional RNABiologyConstants; uses default if None.

        Returns:
            Non-negative penalty in kcal/mol.
        """
        cfg = config or self._bio_config
        in_c3endo = cfg.c3endo_delta_min <= delta_deg <= cfg.c3endo_delta_max
        in_c2endo = cfg.c2endo_delta_min <= delta_deg <= cfg.c2endo_delta_max

        if in_c3endo:
            return 0.0
        if in_c2endo:
            # C2'-endo is OK at non-canonical positions, small penalty in helix
            return 0.0 if position_type == 'non_canonical' else 0.5
        # Outside both windows — strong penalty
        nearest_boundary = min(
            abs(delta_deg - cfg.c3endo_delta_min),
            abs(delta_deg - cfg.c3endo_delta_max),
            abs(delta_deg - cfg.c2endo_delta_min),
            abs(delta_deg - cfg.c2endo_delta_max),
        )
        return 0.1 * nearest_boundary ** 2

    def chi_syn_penalty(self, chi_deg: float) -> float:
        """Penalise syn chi conformation (|chi| < 90 degrees).

        Syn chi is rare in RNA and marks structural distortions.
        Exceptions: first position of kink-turns, Z-form guanines.
        Apply penalty universally; caller may zero it out at known syn sites.

        Args:
            chi_deg: χ torsion angle in degrees.

        Returns:
            Non-negative penalty in kcal/mol.
        """
        if abs(chi_deg) > 90.0:
            return 0.0   # anti conformation — normal
        # Quadratic penalty centred at chi=0 (deepest syn), zero at ±90
        return 0.02 * (90.0 - abs(chi_deg)) ** 2
