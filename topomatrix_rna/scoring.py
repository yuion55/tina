"""Scoring metrics for TOPOMATRIX-RNA pipeline.

Implements TM-score, RMSD, and Wasserstein distance metrics for evaluating
RNA 3D structure predictions.

References:
    [1] Zhang Y, Skolnick J (2004). Proteins 57:702-710 — TM-score
    [2] Kabsch W (1976). Acta Cryst A32:922-923 — optimal rotation
    [3] Cohen-Steiner D et al. (2007). Discrete Comput Geom 37:103-120 — stability
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog

from .data_utils import kabsch_align
from .numba_kernels import tm_score_kernel, wasserstein2_persistence

logger = structlog.get_logger(__name__)


def compute_tm_score(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
    L_target: Optional[int] = None,
) -> float:
    r"""Compute TM-score between predicted and true C3' coordinates.

    Applies Kabsch alignment before scoring.

    Mathematical Basis:
        .. math::
            \text{TM} = \max_{\text{rotation}} \frac{1}{L_{\text{target}}}
            \sum_{i=1}^{L} \frac{1}{1 + (d_i / d_0)^2}

        where :math:`d_0 = 1.24 (L-15)^{1/3} - 1.8` for L > 21, else 0.5.

    Args:
        coords_pred: Predicted C3' coordinates, shape (L, 3).
        coords_true: True C3' coordinates, shape (L, 3).
        L_target: Target length for normalization (defaults to len(coords_true)).

    Returns:
        TM-score in [0, 1].
    """
    if coords_pred.shape[0] == 0 or coords_true.shape[0] == 0:
        return 0.0

    L = min(coords_pred.shape[0], coords_true.shape[0])
    if L_target is None:
        L_target = coords_true.shape[0]

    # Truncate to common length
    pred = coords_pred[:L].copy()
    true = coords_true[:L].copy()

    # Kabsch alignment
    aligned_pred, _, _ = kabsch_align(pred, true)

    return float(tm_score_kernel(aligned_pred, true, L_target))


def compute_rmsd(coords_pred: np.ndarray, coords_true: np.ndarray) -> float:
    """Compute RMSD after Kabsch alignment.

    Args:
        coords_pred: Predicted coordinates, shape (L, 3).
        coords_true: True coordinates, shape (L, 3).

    Returns:
        RMSD value (float >= 0).
    """
    if coords_pred.shape[0] == 0 or coords_true.shape[0] == 0:
        return float("inf")

    L = min(coords_pred.shape[0], coords_true.shape[0])
    _, _, rmsd = kabsch_align(coords_pred[:L], coords_true[:L])
    return float(rmsd)


def compute_gdt_ts(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
    thresholds: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> float:
    """Compute GDT-TS (Global Distance Test - Total Score).

    Args:
        coords_pred: Predicted coordinates, shape (L, 3).
        coords_true: True coordinates, shape (L, 3).
        thresholds: Distance thresholds in Angstroms.

    Returns:
        GDT-TS score in [0, 1].
    """
    if coords_pred.shape[0] == 0 or coords_true.shape[0] == 0:
        return 0.0

    L = min(coords_pred.shape[0], coords_true.shape[0])
    aligned_pred, _, _ = kabsch_align(coords_pred[:L], coords_true[:L])
    distances = np.sqrt(np.sum((aligned_pred - coords_true[:L]) ** 2, axis=1))

    score = 0.0
    for t in thresholds:
        score += np.sum(distances < t) / L
    return score / len(thresholds)


def wasserstein2_diagrams(
    diagram1: np.ndarray, diagram2: np.ndarray
) -> float:
    """Compute Wasserstein-2 distance between two persistence diagrams.

    Args:
        diagram1: First diagram, shape (n1, 2) with columns (birth, death).
        diagram2: Second diagram, shape (n2, 2) with columns (birth, death).

    Returns:
        W2 distance (float >= 0).
    """
    if diagram1.shape[0] == 0 and diagram2.shape[0] == 0:
        return 0.0

    b1 = diagram1[:, 0] if diagram1.shape[0] > 0 else np.empty(0)
    d1 = diagram1[:, 1] if diagram1.shape[0] > 0 else np.empty(0)
    b2 = diagram2[:, 0] if diagram2.shape[0] > 0 else np.empty(0)
    d2 = diagram2[:, 1] if diagram2.shape[0] > 0 else np.empty(0)

    return float(wasserstein2_persistence(b1, d1, b2, d2))


def evaluate_predictions(
    predictions: dict,
    ground_truth: dict,
    sequence_lengths: Optional[dict] = None,
) -> dict:
    """Evaluate a batch of predictions against ground truth.

    Args:
        predictions: Dict mapping seq_id to list of coordinate arrays (5 predictions).
        ground_truth: Dict mapping seq_id to coordinate array.
        sequence_lengths: Optional dict mapping seq_id to target length.

    Returns:
        Dict with per-sequence and aggregate metrics.
    """
    results = {}
    tm_scores = []
    rmsds = []

    for seq_id, pred_list in predictions.items():
        if seq_id not in ground_truth:
            logger.warning("missing_ground_truth", seq_id=seq_id)
            continue

        true_coords = ground_truth[seq_id]
        L_target = true_coords.shape[0]
        if sequence_lengths and seq_id in sequence_lengths:
            L_target = sequence_lengths[seq_id]

        best_tm = 0.0
        best_rmsd = float("inf")

        for pred_coords in pred_list:
            tm = compute_tm_score(pred_coords, true_coords, L_target)
            rmsd = compute_rmsd(pred_coords, true_coords)
            if tm > best_tm:
                best_tm = tm
                best_rmsd = rmsd

        results[seq_id] = {"tm_score": best_tm, "rmsd": best_rmsd, "L": L_target}
        tm_scores.append(best_tm)
        rmsds.append(best_rmsd)

    if tm_scores:
        results["_aggregate"] = {
            "mean_tm_score": float(np.mean(tm_scores)),
            "median_tm_score": float(np.median(tm_scores)),
            "mean_rmsd": float(np.mean(rmsds)),
            "n_evaluated": len(tm_scores),
        }
        logger.info(
            "evaluation_complete",
            n_sequences=len(tm_scores),
            mean_tm=float(np.mean(tm_scores)),
            median_tm=float(np.median(tm_scores)),
        )

    return results
