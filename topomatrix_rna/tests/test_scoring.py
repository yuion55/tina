"""Tests for scoring module (TM-score, RMSD, Wasserstein metrics)."""

import numpy as np
import pytest

from topomatrix_rna.numba_kernels import tm_score_kernel
from topomatrix_rna.scoring import (
    compute_gdt_ts,
    compute_rmsd,
    compute_tm_score,
    evaluate_predictions,
    wasserstein2_diagrams,
)


class TestTMScore:
    """Tests for TM-score computation."""

    def test_perfect_score(self):
        """Identical coordinates should give TM-score of 1.0."""
        coords = np.random.randn(50, 3) * 10
        tm = compute_tm_score(coords, coords)
        assert tm == pytest.approx(1.0, abs=0.01)

    def test_zero_length(self):
        """Empty coordinates should give 0."""
        coords = np.empty((0, 3))
        tm = compute_tm_score(coords, coords)
        assert tm == pytest.approx(0.0)

    def test_tm_kernel_d0(self):
        """Test d0 calculation for different lengths."""
        coords = np.random.randn(30, 3)
        # d0 = 1.24*(30-15)^(1/3) - 1.8 ≈ 1.24*2.466 - 1.8 ≈ 1.258
        tm = tm_score_kernel(coords, coords, 30)
        assert tm == pytest.approx(1.0, abs=0.01)

    def test_tm_kernel_short(self):
        """Short sequences (L<=21) use d0=0.5."""
        coords = np.random.randn(15, 3)
        tm = tm_score_kernel(coords, coords, 15)
        assert tm == pytest.approx(1.0, abs=0.01)

    def test_tm_score_range(self):
        """TM-score should be in [0, 1]."""
        pred = np.random.randn(40, 3) * 10
        true = np.random.randn(40, 3) * 10
        tm = compute_tm_score(pred, true)
        assert 0.0 <= tm <= 1.0


class TestRMSD:
    """Tests for RMSD computation."""

    def test_identical(self):
        """RMSD of identical structures is 0."""
        coords = np.random.randn(30, 3)
        rmsd = compute_rmsd(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_translated(self):
        """RMSD after alignment of translated structure is 0."""
        coords = np.random.randn(30, 3)
        shifted = coords + np.array([10.0, 20.0, 30.0])
        rmsd = compute_rmsd(shifted, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-4)


class TestGDTTS:
    """Tests for GDT-TS."""

    def test_perfect(self):
        """Identical coordinates give GDT-TS of 1.0."""
        coords = np.random.randn(30, 3)
        gdt = compute_gdt_ts(coords, coords)
        assert gdt == pytest.approx(1.0, abs=0.01)


class TestWasserstein2Diagrams:
    """Tests for Wasserstein-2 on persistence diagrams."""

    def test_identical(self):
        """Identical diagrams have zero distance."""
        d = np.array([[0.0, 1.0], [1.0, 3.0]])
        assert wasserstein2_diagrams(d, d) == pytest.approx(0.0, abs=1e-10)

    def test_empty(self):
        """Empty diagrams have zero distance."""
        d = np.empty((0, 2))
        assert wasserstein2_diagrams(d, d) == pytest.approx(0.0)


class TestEvaluatePredictions:
    """Tests for batch evaluation."""

    def test_basic_evaluation(self):
        """Should evaluate predictions and return metrics."""
        coords = np.random.randn(20, 3) * 5
        predictions = {"seq1": [coords, coords + 0.1]}
        ground_truth = {"seq1": coords}

        results = evaluate_predictions(predictions, ground_truth)
        assert "seq1" in results
        assert "_aggregate" in results
        assert results["seq1"]["tm_score"] > 0
