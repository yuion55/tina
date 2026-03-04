"""Tests for Stage 6: TDA Verification Feedback Loop."""

import numpy as np
import pytest

from topomatrix_rna.config import TDAConfig
from topomatrix_rna.numba_kernels import wasserstein2_persistence
from topomatrix_rna.stage6_tda_verify import TDAVerifier


class TestWasserstein2:
    """Tests for Wasserstein-2 between persistence diagrams."""

    def test_identical_diagrams(self):
        """Distance between identical diagrams is 0."""
        b = np.array([0.0, 1.0, 2.0])
        d = np.array([3.0, 4.0, 5.0])
        w2 = wasserstein2_persistence(b, d, b, d)
        assert w2 == pytest.approx(0.0, abs=1e-10)

    def test_empty_diagrams(self):
        """Distance between empty diagrams is 0."""
        b = np.array([], dtype=np.float64)
        d = np.array([], dtype=np.float64)
        w2 = wasserstein2_persistence(b, d, b, d)
        assert w2 == pytest.approx(0.0)

    def test_one_empty(self):
        """Distance to empty diagram equals total persistence."""
        b1 = np.array([0.0])
        d1 = np.array([2.0])
        b2 = np.array([], dtype=np.float64)
        d2 = np.array([], dtype=np.float64)
        w2 = wasserstein2_persistence(b1, d1, b2, d2)
        # Cost = (2-0)^2/4 = 1.0, sqrt(1.0) = 1.0
        assert w2 == pytest.approx(1.0)

    def test_symmetry(self):
        """Distance should be symmetric."""
        b1 = np.array([0.0, 1.0])
        d1 = np.array([3.0, 4.0])
        b2 = np.array([0.5, 1.5])
        d2 = np.array([2.5, 3.5])
        w2_forward = wasserstein2_persistence(b1, d1, b2, d2)
        w2_backward = wasserstein2_persistence(b2, d2, b1, d1)
        assert w2_forward == pytest.approx(w2_backward, abs=0.5)


class TestTDAVerifier:
    """Tests for TDA verification loop."""

    def test_verify_passing(self):
        """Should pass when persistence matches."""
        config = TDAConfig(wasserstein_epsilon=100.0, max_retries=2)
        verifier = TDAVerifier(config)

        theta = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.zeros(10, dtype=np.int64)
        target_bd = np.array([[0.0, 1.0]])

        def compute_pers(t):
            return np.array([0.0]), np.array([1.0])

        def refine(t):
            return t

        best_theta, passed, n_attempts = verifier.verify_and_refine(
            theta, seq, target_bd, compute_pers, refine
        )
        assert passed
        assert n_attempts == 1

    def test_geodesic_perturb(self):
        """Geodesic perturbation should change angles."""
        config = TDAConfig()
        verifier = TDAVerifier(config)

        theta = np.ones((5, 7))
        perturbed = verifier._geodesic_perturb(theta)
        assert perturbed.shape == theta.shape
        assert not np.allclose(perturbed, theta)

    def test_tda_check_passes_when_consistent(self):
        """Large epsilon should allow the check to pass on the first attempt."""
        config = TDAConfig(wasserstein_epsilon=1e6, max_retries=3)
        verifier = TDAVerifier(config, seed=0)

        theta = np.random.uniform(0, 2 * np.pi, (8, 7))
        seq = np.zeros(8, dtype=np.int64)
        target_bd = np.array([[0.0, 0.5], [0.1, 0.8]])

        def compute_pers(t):
            return np.array([0.0, 0.1]), np.array([0.5, 0.8])

        def refine(t):
            return t

        _, passed, n_attempts = verifier.verify_and_refine(
            theta, seq, target_bd, compute_pers, refine
        )
        assert passed
        assert n_attempts == 1

    def test_perturbation_changes_theta(self):
        """Perturbed angles must differ from original."""
        config = TDAConfig(geodesic_kick_scale=0.5)
        verifier = TDAVerifier(config, seed=1)

        theta = np.full((6, 7), 1.0)
        perturbed = verifier._geodesic_perturb(theta)
        assert not np.allclose(perturbed, theta)

    def test_perturbation_reproducible(self):
        """Same seed must produce the same perturbation."""
        config = TDAConfig()
        v1 = TDAVerifier(config, seed=99)
        v2 = TDAVerifier(config, seed=99)

        theta = np.ones((5, 7))
        p1 = v1._geodesic_perturb(theta)
        p2 = v2._geodesic_perturb(theta)
        np.testing.assert_allclose(p1, p2)
