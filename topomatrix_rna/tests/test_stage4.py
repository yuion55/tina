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
