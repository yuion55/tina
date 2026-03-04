"""Tests for Stage 5: Reeb Graph Basin Enumeration."""

import numpy as np
import pytest

from topomatrix_rna.stage5_reeb import ReebBasinEnumerator, ReebGraph


class TestReebGraph:
    """Tests for Reeb graph construction."""

    def test_simple_graph(self):
        """Build Reeb graph from simple energy landscape."""
        energies = np.array([1.0, 3.0, 0.5, 2.0, 4.0])
        adj = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ], dtype=np.float64)

        reeb = ReebGraph()
        reeb.build_from_energy(energies, adj)
        assert len(reeb.nodes) > 0

    def test_get_basins(self):
        """Should extract basins from Reeb graph."""
        energies = np.array([1.0, 5.0, 0.0, 3.0, 2.0])
        adj = np.ones((5, 5)) - np.eye(5)

        reeb = ReebGraph()
        reeb.build_from_energy(energies, adj)
        basins = reeb.get_basins(3)
        assert len(basins) > 0


class TestReebBasinEnumerator:
    """Tests for basin enumeration."""

    def test_enumerate_basic(self):
        """Should select basins from candidates."""
        enumerator = ReebBasinEnumerator(n_basins=3)
        energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        coords = [np.random.randn(10, 3) * (i + 1) for i in range(5)]

        selected = enumerator.enumerate(energies, coords)
        assert len(selected) <= 3
        assert all(0 <= idx < 5 for idx in selected)

    def test_enumerate_fewer_than_requested(self):
        """Should return all candidates if fewer than n_basins."""
        enumerator = ReebBasinEnumerator(n_basins=10)
        energies = np.array([1.0, 2.0])
        coords = [np.random.randn(5, 3) for _ in range(2)]

        selected = enumerator.enumerate(energies, coords)
        assert len(selected) == 2
