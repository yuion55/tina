"""Tests for Stage 2: Hierarchical Tropical Geometry."""

import numpy as np
import pytest

from topomatrix_rna.config import TropicalConfig
from topomatrix_rna.numba_kernels import tropical_gaussian_elim, tropical_min_plus
from topomatrix_rna.stage2_tropical import TropicalBasinCensus


class TestTropicalMinPlus:
    """Tests for tropical (min-plus) matrix multiplication."""

    def test_identity(self):
        """Tropical identity: zeros on diagonal, inf elsewhere."""
        I = np.array([[0.0, np.inf], [np.inf, 0.0]])
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        C = tropical_min_plus(I, A)
        np.testing.assert_allclose(C, A)

    def test_basic_multiplication(self):
        """Basic 2x2 tropical multiplication."""
        A = np.array([[1.0, 3.0], [2.0, 4.0]])
        B = np.array([[0.0, 1.0], [2.0, 0.0]])
        C = tropical_min_plus(A, B)
        # C[0,0] = min(1+0, 3+2) = 1
        # C[0,1] = min(1+1, 3+0) = 2
        # C[1,0] = min(2+0, 4+2) = 2
        # C[1,1] = min(2+1, 4+0) = 3
        expected = np.array([[1.0, 2.0], [2.0, 3.0]])
        np.testing.assert_allclose(C, expected)

    def test_inf_propagation(self):
        """Inf should propagate correctly."""
        A = np.array([[np.inf, 1.0], [2.0, np.inf]])
        B = np.array([[1.0, np.inf], [np.inf, 2.0]])
        C = tropical_min_plus(A, B)
        # C[0,0] = min(inf+1, 1+inf) = inf
        # C[0,1] = min(inf+inf, 1+2) = 3
        assert C[0, 0] == np.inf
        assert C[0, 1] == pytest.approx(3.0)


class TestTropicalGaussianElim:
    """Tests for tropical Gaussian elimination."""

    def test_simple_system(self):
        """Solve simple tropical linear system."""
        A = np.array([[0.0, 1.0], [2.0, 0.0]])
        b = np.array([1.0, 2.0])
        x = tropical_gaussian_elim(A, b)
        # x[0] = min(1-0, 2-2) = 0
        # x[1] = min(1-1, 2-0) = 0
        assert x.shape == (2,)
        assert x[0] == pytest.approx(0.0)
        assert x[1] == pytest.approx(0.0)


class TestTropicalBasinCensus:
    """Tests for the tropical basin census."""

    def test_find_basins_basic(self):
        """Should find at least one basin for simple sequence."""
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        contact_map = np.random.rand(20, 20)
        contact_map = (contact_map + contact_map.T) / 2
        basins = census.find_basins("GGGGAAAACCCCUUUUAAAA", contact_map, n_basins=3)
        assert len(basins) >= 1
        for basin in basins:
            assert basin.ndim == 2

    def test_find_basins_short(self):
        """Short sequence should still return a basin."""
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        basins = census.find_basins("ACGU", np.zeros((4, 4)), n_basins=1)
        assert len(basins) >= 1

    def test_find_basins_empty(self):
        """Very short sequence returns empty basin."""
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        basins = census.find_basins("AC", np.zeros((2, 2)), n_basins=1)
        assert len(basins) >= 1


class TestWeightMatrixBiology:
    """Tests for biologically corrected weight matrix values."""

    def test_gc_weight_correct(self):
        """G-C pair should use weight 1.0."""
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        # Encoding: A=0, C=1, G=2, U=3
        encoded = np.array([2, 1], dtype=np.int64)  # G-C
        cm = np.ones((2, 2))
        W = census._compute_weight_matrix(encoded, cm)
        # The bp_weight for G-C is config.weight_bp * 1.0 = -2.0 * 1.0 = -2.0
        # No stacking at L=2 and loop constraint k>=4 prevents use, so these
        # should remain inf at L=2 (loop length < 4).
        # Test with longer sequence instead.
        encoded = np.array([2, 0, 0, 0, 0, 1], dtype=np.int64)  # G....C
        cm = np.ones((6, 6))
        W = census._compute_weight_matrix(encoded, cm)
        # W[0, 5] should be finite (G-C pair with loop length 4)
        assert np.isfinite(W[0, 5])

    def test_gu_wobble_weight(self):
        """G-U wobble pair should use weight 0.7."""
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        encoded = np.array([2, 0, 0, 0, 0, 3], dtype=np.int64)  # G....U
        cm = np.ones((6, 6))
        W = census._compute_weight_matrix(encoded, cm)
        # W[0,5] should be config.weight_bp * 0.7 * (1 + cm[0,5])
        expected = config.weight_bp * 0.7 * (1.0 + cm[0, 5])
        assert W[0, 5] == pytest.approx(expected)


class TestElectrostaticPenalty:
    """Tests for Debye-Hückel electrostatic penalty."""

    def test_basic_penalty(self):
        """Penalty should be non-negative and finite."""
        from topomatrix_rna.config import RNABiologyConstants
        from topomatrix_rna.stage2_tropical import compute_electrostatic_penalty
        coords = np.random.randn(10, 3) * 5.0
        config = RNABiologyConstants()
        penalty = compute_electrostatic_penalty(coords, config)
        assert penalty.shape == (10, 10)
        assert np.all(penalty >= 0)
        assert np.all(np.isfinite(penalty))

    def test_distant_atoms_zero_penalty(self):
        """Distant atoms should have zero penalty."""
        from topomatrix_rna.config import RNABiologyConstants
        from topomatrix_rna.stage2_tropical import compute_electrostatic_penalty
        coords = np.zeros((5, 3))
        for i in range(5):
            coords[i, 0] = i * 100.0  # 100 Å apart
        config = RNABiologyConstants()
        penalty = compute_electrostatic_penalty(coords, config)
        assert np.allclose(penalty, 0.0)


class TestPseudotorsions:
    """Tests for η/θ pseudotorsion computation."""

    def test_basic_pseudotorsions(self):
        """Pseudotorsions should return correct shapes with NaN at termini."""
        from topomatrix_rna.stage2_tropical import compute_pseudotorsions
        L = 10
        c4 = np.random.randn(L, 3)
        p = np.random.randn(L, 3)
        eta, theta = compute_pseudotorsions(c4, p)
        assert eta.shape == (L,)
        assert theta.shape == (L,)
        assert np.isnan(eta[0])
        assert np.isnan(eta[-1])
        assert np.isnan(theta[0])
        assert np.isnan(theta[-1])

    def test_interior_values_finite(self):
        """Interior pseudotorsion values should be finite."""
        from topomatrix_rna.stage2_tropical import compute_pseudotorsions
        L = 10
        c4 = np.random.randn(L, 3) * 5.0
        p = np.random.randn(L, 3) * 5.0
        eta, theta = compute_pseudotorsions(c4, p)
        for i in range(1, L - 1):
            assert np.isfinite(eta[i])
            assert np.isfinite(theta[i])


class TestTropicalDPMinSpan:
    """Tests for DP handling of minimum separation base pairs (j-i=4)."""

    def test_dp_finds_pair_at_min_separation(self):
        """DP should find base pairs at exactly j-i=4 (minimum separation).

        Regression test: DP previously started at span=5 while the weight
        matrix allowed pairs at j-i >= 4, causing pairs at j-i=4 to be
        missed entirely.
        """
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        # Sequence: G....C  (length 6, G at 0, C at 5)
        # Pair (0, 5) has j-i=5, always found
        # Sequence: G...C   (length 5, G at 0, C at 4)
        # Pair (0, 4) has j-i=4, was previously missed
        seq_5 = "GAAAC"
        cm_5 = np.ones((5, 5))
        basins = census.find_basins(seq_5, cm_5, n_basins=1)
        # Should find at least one basin with a pair at (0, 4)
        assert len(basins) >= 1
        found_pair = False
        for basin in basins:
            if basin.shape[0] > 0:
                for row in range(basin.shape[0]):
                    if basin[row, 0] == 0 and basin[row, 1] == 4:
                        found_pair = True
        assert found_pair, "DP should find pair (0, 4) at minimum separation j-i=4"

    def test_dp_l5_gc_pair(self):
        """Length-5 G-C sequence should produce a base pair."""
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        seq = "GUUAC"  # G(0) can pair with C(4), j-i=4
        cm = np.ones((5, 5))
        W = census._compute_weight_matrix(
            np.array([2, 3, 3, 0, 1], dtype=np.int64), cm
        )
        # W[0, 4] should be finite (G-C pair)
        assert np.isfinite(W[0, 4]), "Weight matrix should allow pair at j-i=4"
        pairs = census._tropical_dp(W, 5)
        # The DP should find this pair
        assert any(p == (0, 4) for p in pairs), (
            "DP should find pair (0, 4) for length-5 G-C sequence"
        )
