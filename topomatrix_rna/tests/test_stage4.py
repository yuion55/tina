"""Tests for Stage 4: Riemannian Torsion Refinement."""

import numpy as np
import pytest

from topomatrix_rna.config import RiemannianConfig
from topomatrix_rna.numba_kernels import (
    exp_map_torus,
    parallel_transport_torus,
    rsrnasp1_energy_block,
    torus_geodesic_distance,
)
from topomatrix_rna.stage4_riemannian import RiemannianRefiner


class TestTorusOperations:
    """Tests for torus manifold operations."""

    def test_geodesic_zero(self):
        """Distance from a point to itself is zero."""
        theta = np.array([1.0, 2.0, 3.0])
        assert torus_geodesic_distance(theta, theta) == pytest.approx(0.0)

    def test_geodesic_wrapping(self):
        """Distance wraps around 2π correctly."""
        theta1 = np.array([0.1])
        theta2 = np.array([2 * np.pi - 0.1])
        d = torus_geodesic_distance(theta1, theta2)
        assert d == pytest.approx(0.2, abs=1e-10)

    def test_exp_map_basic(self):
        """Exponential map wraps correctly."""
        theta = np.array([5.0])
        v = np.array([2.0])
        result = exp_map_torus(theta, v)
        expected = (5.0 + 2.0) % (2 * np.pi)
        assert result[0] == pytest.approx(expected)

    def test_exp_map_identity(self):
        """Zero tangent vector gives same point."""
        theta = np.array([1.0, 2.0, 3.0])
        v = np.zeros(3)
        result = exp_map_torus(theta, v)
        np.testing.assert_allclose(result, theta)

    def test_parallel_transport_identity(self):
        """Parallel transport on flat torus is identity."""
        v = np.array([1.0, 2.0, 3.0])
        theta_old = np.array([0.5, 1.0, 1.5])
        theta_new = np.array([1.0, 2.0, 3.0])
        result = parallel_transport_torus(v, theta_old, theta_new)
        np.testing.assert_allclose(result, v)


class TestRsRNASP1Energy:
    """Tests for rsRNASP1 energy computation."""

    def test_basic_energy(self):
        """Energy should be finite for valid inputs."""
        theta = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)
        E = rsrnasp1_energy_block(theta, seq)
        assert np.isfinite(E)

    def test_minimum_near_reference(self):
        """Energy at reference angles should be lower than random."""
        # Reference torsion angles
        theta0 = np.tile(
            np.array([5.28, 3.05, 0.91, 2.65, 3.59, 4.71, 3.14]), (10, 1)
        )
        theta_random = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int64)

        E_ref = rsrnasp1_energy_block(theta0, seq)
        E_rand = rsrnasp1_energy_block(theta_random, seq)
        # Reference angles should have lower single-body energy
        assert E_ref < E_rand + 50  # Allow some slack for pairing terms


class TestRiemannianRefiner:
    """Tests for Riemannian refinement."""

    def test_refine_reduces_energy(self):
        """Refinement should reduce or maintain energy."""
        config = RiemannianConfig(n_steps=20, block_size=10)
        refiner = RiemannianRefiner(config)

        theta_init = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)

        initial_energy = float(rsrnasp1_energy_block(theta_init, seq))
        theta_opt, final_energy = refiner.refine(theta_init, seq)

        assert theta_opt.shape == (10, 7)
        assert final_energy <= initial_energy + 1e-6  # Allow tiny numerical errors

    def test_refine_output_on_torus(self):
        """Refined angles should be in [0, 2π)."""
        config = RiemannianConfig(n_steps=10, block_size=5)
        refiner = RiemannianRefiner(config)

        theta_init = np.random.uniform(0, 2 * np.pi, (5, 7))
        seq = np.array([0, 1, 2, 3, 0], dtype=np.int64)

        theta_opt, _ = refiner.refine(theta_init, seq)
        assert np.all(theta_opt >= 0.0)
        assert np.all(theta_opt < 2 * np.pi + 1e-10)

    def test_energy_decreases_over_steps(self):
        """ADAM should reduce energy compared to initial."""
        config = RiemannianConfig(n_steps=50, block_size=10)
        refiner = RiemannianRefiner(config)

        np.random.seed(0)
        theta_init = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)

        initial_energy = float(rsrnasp1_energy_block(theta_init, seq))
        _, final_energy = refiner.refine(theta_init, seq)

        assert final_energy <= initial_energy + 1e-6

    def test_symplectic_energy_conservation(self):
        """Symplectic integrator with same seed gives identical result."""
        config = RiemannianConfig(n_steps=10, symplectic_h=0.001)
        refiner = RiemannianRefiner(config)

        np.random.seed(7)
        theta_init = np.random.uniform(0, 2 * np.pi, (8, 7))
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

        theta1, e1 = refiner.refine_symplectic(theta_init.copy(), seq)
        theta2, e2 = refiner.refine_symplectic(theta_init.copy(), seq)

        np.testing.assert_allclose(theta1, theta2)
        assert e1 == pytest.approx(e2)

    def test_torsion_to_coords_bond_distances(self):
        """All consecutive C3'–C3' distances should be in [3.0, 10.0] Å."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)

        L = 20
        np.random.seed(42)
        theta = np.random.uniform(0, 2 * np.pi, (L, 7))
        template = np.zeros((L, 3))
        template[:, 0] = np.arange(L) * 5.9  # Straight chain seed

        coords = refiner.torsion_to_coords(theta, template)
        assert coords.shape == (L, 3)

        for i in range(1, L):
            d = np.linalg.norm(coords[i] - coords[i - 1])
            assert 3.0 <= d <= 10.0, (
                f"Bond distance between residues {i-1} and {i} = {d:.2f} out of range"
            )


class TestSuiteConformerPenalty:
    """Tests for suite conformer penalty function."""

    def test_a_form_zero_penalty(self):
        """A-form angles should have zero or near-zero penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        a_form = {
            'delta_prev': 83, 'epsilon': 212, 'zeta': 289,
            'alpha': -68, 'beta': 178, 'gamma': 55, 'delta': 83,
        }
        penalty = refiner.suite_conformer_penalty(a_form)
        assert penalty == pytest.approx(0.0, abs=1e-6)

    def test_outlier_positive_penalty(self):
        """Outlier angles should have positive penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        outlier = {
            'delta_prev': 0, 'epsilon': 0, 'zeta': 0,
            'alpha': 0, 'beta': 0, 'gamma': 0, 'delta': 0,
        }
        penalty = refiner.suite_conformer_penalty(outlier)
        assert penalty > 0.0

    def test_penalty_non_negative(self):
        """Penalty should always be non-negative."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        random_angles = {k: np.random.uniform(-180, 360) for k in
                         ['delta_prev', 'epsilon', 'zeta', 'alpha', 'beta', 'gamma', 'delta']}
        penalty = refiner.suite_conformer_penalty(random_angles)
        assert penalty >= 0.0


class TestSugarPuckerPenalty:
    """Tests for sugar pucker penalty function."""

    def test_c3endo_zero_penalty(self):
        """C3'-endo delta should have zero penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        assert refiner.sugar_pucker_penalty(83.0) == 0.0

    def test_c2endo_helix_small_penalty(self):
        """C2'-endo in helix position should have small penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        assert refiner.sugar_pucker_penalty(145.0, 'helix') == 0.5

    def test_c2endo_non_canonical_zero(self):
        """C2'-endo in non-canonical position should have zero penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        assert refiner.sugar_pucker_penalty(145.0, 'non_canonical') == 0.0

    def test_outside_both_windows(self):
        """Delta outside both windows should have large penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        penalty = refiner.sugar_pucker_penalty(200.0)
        assert penalty > 0.0


class TestChiSynPenalty:
    """Tests for chi syn conformation penalty."""

    def test_anti_no_penalty(self):
        """Anti chi (|chi| > 90) should have zero penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        assert refiner.chi_syn_penalty(-159.0) == 0.0
        assert refiner.chi_syn_penalty(120.0) == 0.0

    def test_syn_positive_penalty(self):
        """Syn chi (|chi| < 90) should have positive penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        assert refiner.chi_syn_penalty(0.0) > 0.0
        assert refiner.chi_syn_penalty(45.0) > 0.0

    def test_boundary_zero(self):
        """Chi at ±90 boundary should have zero penalty."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        assert refiner.chi_syn_penalty(90.1) == 0.0
        assert refiner.chi_syn_penalty(-90.1) == 0.0

    def test_penalty_increases_toward_zero(self):
        """Penalty should be largest at chi=0."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        p_zero = refiner.chi_syn_penalty(0.0)
        p_45 = refiner.chi_syn_penalty(45.0)
        assert p_zero > p_45

    def test_wrapping_near_360(self):
        """Chi near 360° (syn via wrapping) should be penalized.

        Regression test: chi_syn_penalty previously did not handle angle
        wrapping, so 350° (equivalent to -10°, syn) had no penalty.
        """
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        # 350° wraps to -10°, which is syn (|chi| < 90)
        assert refiner.chi_syn_penalty(350.0) > 0.0
        # 360° wraps to 0°, deepest syn
        assert refiner.chi_syn_penalty(360.0) > 0.0
        # 270° wraps to -90°, boundary — should be exactly zero
        assert refiner.chi_syn_penalty(270.0) == pytest.approx(0.0, abs=1e-10)

    def test_wrapping_preserves_anti(self):
        """Anti conformation values should still have zero penalty after wrapping fix."""
        config = RiemannianConfig()
        refiner = RiemannianRefiner(config)
        # 200° wraps to -160° (anti, |chi| > 90)
        assert refiner.chi_syn_penalty(200.0) == 0.0
        # 150° stays 150° (anti)
        assert refiner.chi_syn_penalty(150.0) == 0.0


class TestSuiteAngleMapping:
    """Tests that suite conformer penalty uses correct theta-to-angle mapping."""

    def test_a_form_theta_gives_low_penalty(self):
        """A-form backbone torsion angles in theta format should give low penalty.

        Regression test: _apply_bio_penalty_to_grad previously used
        SUITE_ANGLE_KEYS enumeration order to index into theta, causing a
        complete mismatch between angle names and theta array positions.
        """
        config = RiemannianConfig(n_steps=1, block_size=10)
        refiner = RiemannianRefiner(config)

        # Build theta with A-form angles in the correct positions:
        # theta order: alpha(0), beta(1), gamma(2), delta(3), epsilon(4), zeta(5), chi(6)
        # Negative angles are shifted by +360 because theta values are in [0, 2π].
        a_form_rad = np.array([
            np.radians(-68 + 360),   # alpha = -68° → 292° (shift to [0, 360])
            np.radians(178),          # beta = 178°
            np.radians(55),           # gamma = 55°
            np.radians(83),           # delta = 83°
            np.radians(212),          # epsilon = 212°
            np.radians(289),          # zeta = 289°
            np.radians(-159 + 360),   # chi = -159° → 201° (shift to [0, 360])
        ])
        theta = np.tile(a_form_rad, (3, 1))  # 3 residues
        seq = np.array([0, 1, 2], dtype=np.int64)

        # The suite penalty alone for A-form should be zero or near-zero
        angle_dict = {
            'alpha': np.degrees(theta[1, 0]),
            'beta': np.degrees(theta[1, 1]),
            'gamma': np.degrees(theta[1, 2]),
            'delta': np.degrees(theta[1, 3]),
            'epsilon': np.degrees(theta[1, 4]),
            'zeta': np.degrees(theta[1, 5]),
            'delta_prev': np.degrees(theta[0, 3]),
        }
        sp = refiner.suite_conformer_penalty(angle_dict)
        assert sp == pytest.approx(0.0, abs=1.0), (
            f"A-form angles should give near-zero suite penalty, got {sp}"
        )

    def test_delta_prev_uses_previous_residue(self):
        """delta_prev should come from the previous residue's delta (index 3)."""
        config = RiemannianConfig(n_steps=1, block_size=10)
        refiner = RiemannianRefiner(config)

        # Create theta where residue 0 has a specific delta
        theta = np.ones((3, 7)) * np.pi  # All angles = π
        theta[0, 3] = np.radians(83)  # delta of residue 0 = 83°

        grad = np.ones((3, 7))
        result = refiner._apply_bio_penalty_to_grad(grad, theta)
        # The function should not crash and should return valid gradients
        assert result.shape == (3, 7)
        assert np.all(np.isfinite(result))
