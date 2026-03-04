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
