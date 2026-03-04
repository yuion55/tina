"""Tests for Stage 1: Contact Map via RG Matrix Field Theory."""

import numpy as np
import pytest

from topomatrix_rna.config import ContactMapConfig
from topomatrix_rna.numba_kernels import rg_block_contact_map
from topomatrix_rna.stage1_contact_map import ContactMapPredictor


class TestRGBlockContactMap:
    """Tests for the RG block contact map kernel."""

    def test_basic_block(self):
        """Block contact map should return valid probabilities."""
        seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)
        P = rg_block_contact_map(seq, 0.3, 0.1, 50, 1e-6)
        assert P.shape == (10, 10)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)

    def test_symmetry(self):
        """Contact map should be symmetric."""
        seq = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        P = rg_block_contact_map(seq, 0.3, 0.1, 50, 1e-6)
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_empty_sequence(self):
        """Empty sequence should return empty map."""
        seq = np.array([], dtype=np.int64)
        P = rg_block_contact_map(seq, 0.3, 0.1, 50, 1e-6)
        assert P.shape == (0, 0)

    def test_short_sequence(self):
        """Very short sequence should still work."""
        seq = np.array([0, 1], dtype=np.int64)
        P = rg_block_contact_map(seq, 0.3, 0.1, 50, 1e-6)
        assert P.shape == (2, 2)


class TestContactMapPredictor:
    """Tests for full contact map prediction."""

    def test_predict_basic(self):
        """Should predict contact map for a simple sequence."""
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        P = predictor.predict("ACGUACGUACGU")
        assert P.shape == (12, 12)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)

    def test_predict_symmetric(self):
        """Contact map should be symmetric."""
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        P = predictor.predict("GGGAAACCC")
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_predict_empty(self):
        """Empty sequence should return empty map."""
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        P = predictor.predict("")
        assert P.shape == (0, 0)

    def test_extract_genus(self):
        """Genus extraction should return non-negative integer."""
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        P = predictor.predict("GGGGAAAACCCC")
        g = predictor.extract_genus(P)
        assert isinstance(g, int)
        assert g >= 0
