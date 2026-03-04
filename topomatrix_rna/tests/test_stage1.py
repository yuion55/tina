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


class TestLWWeights:
    """Tests for Leontis-Westhof base pair weight lookup."""

    def test_gc_cww(self):
        """G-C cWW should return 1.0."""
        from topomatrix_rna.stage1_contact_map import get_lw_weight
        assert get_lw_weight('G', 'C', 'cWW') == 1.0

    def test_au_cww(self):
        """A-U cWW should return 0.9."""
        from topomatrix_rna.stage1_contact_map import get_lw_weight
        assert get_lw_weight('A', 'U', 'cWW') == 0.9

    def test_gu_wobble(self):
        """G-U cWW (wobble) should return 0.7."""
        from topomatrix_rna.stage1_contact_map import get_lw_weight
        assert get_lw_weight('G', 'U', 'cWW') == 0.7

    def test_symmetric_fallback(self):
        """Lookup should fall back to reversed pair."""
        from topomatrix_rna.stage1_contact_map import get_lw_weight
        assert get_lw_weight('C', 'G', 'cWW') == 1.0

    def test_unknown_pair_default(self):
        """Unknown pair should return default 0.4."""
        from topomatrix_rna.stage1_contact_map import get_lw_weight
        assert get_lw_weight('U', 'U', 'cWW') == 0.4

    def test_non_canonical_family(self):
        """Known non-canonical family should return correct weight."""
        from topomatrix_rna.stage1_contact_map import get_lw_weight
        assert get_lw_weight('G', 'A', 'tHS') == 0.5


class TestDetectTetraloops:
    """Tests for GNRA/UNCG tetraloop detection."""

    def test_gnra(self):
        """Should detect GNRA tetraloops."""
        from topomatrix_rna.stage1_contact_map import detect_tetraloops
        hits = detect_tetraloops("GAAAGAAA")
        assert 0 in hits
        assert hits[0] == 'GNRA'

    def test_uucg(self):
        """Should detect UUCG tetraloop."""
        from topomatrix_rna.stage1_contact_map import detect_tetraloops
        hits = detect_tetraloops("AAUUCGAA")
        assert 2 in hits
        assert hits[2] == 'UUCG'

    def test_uncg(self):
        """Should detect UNCG tetraloop."""
        from topomatrix_rna.stage1_contact_map import detect_tetraloops
        hits = detect_tetraloops("AAUACGAA")
        assert 2 in hits
        assert hits[2] == 'UNCG'

    def test_no_tetraloops(self):
        """Should return empty dict when no tetraloops present."""
        from topomatrix_rna.stage1_contact_map import detect_tetraloops
        hits = detect_tetraloops("CCCCCCCC")
        assert len(hits) == 0

    def test_uucg_preferred_over_uncg(self):
        """UUCG should be preferred over generic UNCG at same position."""
        from topomatrix_rna.stage1_contact_map import detect_tetraloops
        hits = detect_tetraloops("UUCG")
        assert 0 in hits
        assert hits[0] == 'UUCG'


class TestFindCrossingPairs:
    """Tests for pseudoknot crossing pair detection."""

    def test_no_crossing(self):
        """Nested pairs should have no crossings."""
        from topomatrix_rna.stage1_contact_map import find_crossing_pairs
        bp_list = [(0, 10), (1, 9), (2, 8)]
        assert len(find_crossing_pairs(bp_list)) == 0

    def test_simple_crossing(self):
        """Simple pseudoknot should be detected."""
        from topomatrix_rna.stage1_contact_map import find_crossing_pairs
        bp_list = [(0, 5), (3, 8)]  # 0 < 3 < 5 < 8
        crossing = find_crossing_pairs(bp_list)
        assert (3, 8) in crossing

    def test_empty_list(self):
        """Empty input should return empty set."""
        from topomatrix_rna.stage1_contact_map import find_crossing_pairs
        assert len(find_crossing_pairs([])) == 0


class TestDetectCoaxialJunctions:
    """Tests for coaxial stack junction detection."""

    def test_adjacent_helices(self):
        """Adjacent helices (gap=0) should be detected as coaxial."""
        from topomatrix_rna.stage1_contact_map import detect_coaxial_junctions
        helices = [(0, 10), (10, 20)]
        pairs = detect_coaxial_junctions(helices)
        assert (0, 1) in pairs

    def test_gap_one(self):
        """Helices with gap of 1 should be detected."""
        from topomatrix_rna.stage1_contact_map import detect_coaxial_junctions
        helices = [(0, 10), (11, 20)]
        pairs = detect_coaxial_junctions(helices)
        assert (0, 1) in pairs

    def test_no_coaxial(self):
        """Distant helices should not be coaxial."""
        from topomatrix_rna.stage1_contact_map import detect_coaxial_junctions
        helices = [(0, 10), (15, 25)]
        pairs = detect_coaxial_junctions(helices)
        assert len(pairs) == 0
