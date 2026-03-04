"""Tests for Stage 0: Topological Atlas Construction."""

import numpy as np
import pytest

from topomatrix_rna.config import AtlasConfig
from topomatrix_rna.numba_kernels import (
    compute_genus_gauss_code,
    maxmin_landmark_sampling,
    persistence_image_kernel,
    stable_rank_signature,
)
from topomatrix_rna.stage0_atlas import TopologicalAtlas


class TestGenusComputation:
    """Tests for topological genus via Gauss code rank."""

    def test_genus_no_arcs(self):
        """Empty base pair set should give genus 0."""
        arc_i = np.array([], dtype=np.int64)
        arc_j = np.array([], dtype=np.int64)
        assert compute_genus_gauss_code(arc_i, arc_j, 10) == 0

    def test_genus_nested_arcs(self):
        """Nested (non-crossing) arcs give genus 0 (planar)."""
        # Arcs: (0,9), (1,8), (2,7) — nested, no crossings
        arc_i = np.array([0, 1, 2], dtype=np.int64)
        arc_j = np.array([9, 8, 7], dtype=np.int64)
        g = compute_genus_gauss_code(arc_i, arc_j, 10)
        assert g == 0

    def test_genus_crossing_arcs(self):
        """Crossing arcs produce nonzero genus (pseudoknot)."""
        # Arcs: (0,5), (3,8) — these cross: 0 < 3 < 5 < 8
        arc_i = np.array([0, 3], dtype=np.int64)
        arc_j = np.array([5, 8], dtype=np.int64)
        g = compute_genus_gauss_code(arc_i, arc_j, 10)
        assert g >= 1


class TestMaxminLandmarkSampling:
    """Tests for farthest-point landmark sampling."""

    def test_basic_sampling(self):
        """Should select n_landmarks distinct points."""
        np.random.seed(42)
        coords = np.random.randn(100, 3)
        indices = maxmin_landmark_sampling(coords, 10)
        assert indices.shape[0] == 10
        assert len(set(indices)) == 10

    def test_more_landmarks_than_points(self):
        """Should return all points if n_landmarks >= N."""
        coords = np.random.randn(5, 3)
        indices = maxmin_landmark_sampling(coords, 10)
        assert indices.shape[0] == 5

    def test_first_landmark_is_zero(self):
        """First selected landmark should be index 0."""
        coords = np.random.randn(50, 3)
        indices = maxmin_landmark_sampling(coords, 5)
        assert indices[0] == 0


class TestPersistenceImage:
    """Tests for persistence image computation."""

    def test_empty_diagram(self):
        """Empty persistence diagram gives zero image."""
        birth = np.array([], dtype=np.float64)
        death = np.array([], dtype=np.float64)
        grid_x = np.linspace(0, 10, 10)
        grid_y = np.linspace(0, 10, 10)
        img = persistence_image_kernel(birth, death, grid_x, grid_y, 0.2)
        assert img.shape == (10, 10)
        assert np.allclose(img, 0.0)

    def test_single_point(self):
        """Single persistence point gives non-zero image."""
        birth = np.array([0.0])
        death = np.array([5.0])
        grid_x = np.linspace(0, 10, 20)
        grid_y = np.linspace(0, 10, 20)
        img = persistence_image_kernel(birth, death, grid_x, grid_y, 1.0)
        assert img.shape == (20, 20)
        assert np.sum(img) > 0

    def test_output_shape(self):
        """Output shape should match grid dimensions."""
        birth = np.array([0.0, 1.0, 2.0])
        death = np.array([3.0, 4.0, 5.0])
        grid_x = np.linspace(0, 10, 50)
        grid_y = np.linspace(0, 10, 50)
        img = persistence_image_kernel(birth, death, grid_x, grid_y, 0.5)
        assert img.shape == (50, 50)


class TestStableRank:
    """Tests for stable rank signature."""

    def test_basic_signature(self):
        """Stable rank should decrease with increasing threshold."""
        birth = np.array([0.0, 0.0, 0.0, 1.0])
        death = np.array([1.0, 3.0, 5.0, 4.0])
        thresholds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        sr = stable_rank_signature(birth, death, thresholds)
        assert sr.shape == (5,)
        # Should be non-increasing
        for i in range(len(sr) - 1):
            assert sr[i] >= sr[i + 1]


class TestAtlas:
    """Integration tests for TopologicalAtlas."""

    def test_rips_persistence_h0(self):
        """H0 persistence from simple point cloud."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        from scipy.spatial.distance import cdist
        dist = cdist(coords, coords)
        pairs = TopologicalAtlas._rips_persistence_h0(dist)
        assert pairs.shape[1] == 2
        assert pairs.shape[0] == 2  # n-1 merges
        # First merge at distance 1.0
        assert np.min(pairs[:, 1]) == pytest.approx(1.0)

    def test_atlas_init(self):
        """Atlas should initialize with default config."""
        config = AtlasConfig()
        atlas = TopologicalAtlas(config)
        assert len(atlas.entries) == 0
