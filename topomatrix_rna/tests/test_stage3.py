"""Tests for Stage 3: Topological Template Retrieval."""

import numpy as np
import pytest

from topomatrix_rna.config import AtlasConfig, RetrievalConfig
from topomatrix_rna.numba_kernels import sliced_wasserstein_1d
from topomatrix_rna.stage0_atlas import AtlasEntry, TopologicalAtlas
from topomatrix_rna.stage3_retrieval import TemplateRetriever


class TestSlicedWasserstein:
    """Tests for 1D Wasserstein distance."""

    def test_identical(self):
        """Distance between identical sorted arrays is 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert sliced_wasserstein_1d(a, a) == pytest.approx(0.0)

    def test_shifted(self):
        """Distance with uniform shift."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert sliced_wasserstein_1d(a, b) == pytest.approx(1.0)

    def test_empty(self):
        """Empty arrays give zero distance."""
        a = np.array([], dtype=np.float64)
        assert sliced_wasserstein_1d(a, a) == pytest.approx(0.0)


def _make_test_atlas() -> TopologicalAtlas:
    """Create a test atlas with synthetic entries."""
    config = AtlasConfig()
    atlas = TopologicalAtlas(config)

    for i in range(5):
        entry = AtlasEntry(
            pdb_id=f"test_{i}",
            sequence="ACGU" * 10,
            length=40,
            genus=i % 3,
            persistence_image=np.random.rand(50, 50),
            stable_rank=np.random.rand(64),
            coords_c3=np.random.randn(40, 3) * 10,
            birth_death=np.array([[0.0, float(i + 1)] for _ in range(3)]),
        )
        atlas.entries[f"test_{i}"] = entry

    return atlas


class TestTemplateRetriever:
    """Tests for template retrieval."""

    def test_retrieve_basic(self):
        """Should retrieve templates from atlas."""
        config = RetrievalConfig()
        retriever = TemplateRetriever(config)
        atlas = _make_test_atlas()

        query_sr = np.random.rand(64)
        query_bd = np.array([[0.0, 2.0], [0.0, 3.0]])

        results = retriever.retrieve(1, query_sr, query_bd, atlas)
        assert len(results) > 0
        assert len(results) <= config.retrieval_top_k

    def test_retrieve_empty_atlas(self):
        """Should handle empty atlas gracefully."""
        config = RetrievalConfig()
        retriever = TemplateRetriever(config)
        atlas = TopologicalAtlas(AtlasConfig())

        results = retriever.retrieve(0, np.zeros(64), np.empty((0, 2)), atlas)
        assert len(results) == 0

    def test_retrieve_novel(self):
        """Fallback retrieval for novel topologies."""
        config = RetrievalConfig()
        retriever = TemplateRetriever(config)
        atlas = _make_test_atlas()

        results = retriever.retrieve_for_novel_topology(5, atlas)
        assert len(results) > 0
