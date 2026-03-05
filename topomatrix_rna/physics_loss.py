"""7-component physics-informed loss using RNA biology constants from config.py.

Loss Components:
    1. FAPE: Frame Aligned Point Error
    2. Distogram CE: Cross-entropy on distance bins
    3. Bond geometry: Virtual bond length + angle constraints
    4. Clash: Steric clash penalty
    5. Stacking: Consecutive nucleotide stacking reward
    6. H-bond: Watson-Crick base pair distance reward
    7. Suite torsion: Richardson suite conformer penalty

Mathematical Basis:
    :math:`\\mathcal{L} = \\lambda_1 L_{\\text{FAPE}} +
    \\lambda_2 L_{\\text{disto}} + \\lambda_3 L_{\\text{bond}} +
    \\lambda_4 L_{\\text{clash}} + \\lambda_5 L_{\\text{stack}} +
    \\lambda_6 L_{\\text{hbond}} + \\lambda_7 L_{\\text{suite}}`
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch_not_available", msg="PyTorch not installed; loss unavailable")


def coords_to_distogram_target(
    coords: "torch.Tensor", n_bins: int = 64,
    d_min: float = 2.0, d_max: float = 22.0,
) -> "torch.Tensor":
    r"""Bin pairwise distances into a distogram target.

    :math:`\\text{bin}_k = d_{\\min} + k \\cdot \\Delta,\\quad
    \\Delta = (d_{\\max} - d_{\\min}) / n_{\\text{bins}}`

    Args:
        coords: Coordinates, shape ``(L, 3)``.
        n_bins: Number of distance bins.
        d_min: Minimum distance.
        d_max: Maximum distance.

    Returns:
        Binned target, shape ``(L, L)`` with integer bin indices.
    """
    dists = torch.cdist(coords, coords)  # (L, L)
    bin_width = (d_max - d_min) / n_bins
    binned = ((dists - d_min) / bin_width).long().clamp(0, n_bins - 1)
    return binned


if _TORCH_AVAILABLE:

    class PhysicsInformedLoss(nn.Module):
        r"""7-component physics-informed loss for RNA 3D structure prediction.

        Uses RNA biology constants from ``RNABiologyConstants`` in config.py.
        """

        def __init__(self, biology_config: Optional[object] = None) -> None:
            super().__init__()
            from .config import RNABiologyConstants

            self.bio = biology_config if biology_config is not None else RNABiologyConstants()

            # Loss weights (lambdas)
            self.lambda_fape = 1.0
            self.lambda_distogram = 0.3
            self.lambda_bond = 1.0
            self.lambda_clash = 0.5
            self.lambda_stack = 0.2
            self.lambda_hbond = 0.3
            self.lambda_suite = 0.1

        def forward(
            self,
            pred_coords: torch.Tensor,
            true_coords: torch.Tensor,
            pred_distogram: torch.Tensor,
            true_distogram: torch.Tensor,
            sequence: Optional[torch.Tensor] = None,
            frames: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Compute total physics-informed loss.

            Args:
                pred_coords: Predicted C3' coordinates, shape ``(L, 3)``.
                true_coords: True C3' coordinates, shape ``(L, 3)``.
                pred_distogram: Predicted distogram logits, shape ``(L, L, bins)``.
                true_distogram: True distance bin indices, shape ``(L, L)``.
                sequence: Optional encoded sequence, shape ``(L,)``.
                frames: Optional per-residue rigid frames (unused for now).

            Returns:
                Scalar loss value.
            """
            L = pred_coords.shape[0]

            # 1. FAPE loss
            loss_fape = self._fape_loss(pred_coords, true_coords)

            # 2. Distogram cross-entropy
            loss_disto = self._distogram_loss(pred_distogram, true_distogram)

            # 3. Bond geometry
            loss_bond = self._bond_geometry_loss(pred_coords)

            # 4. Clash penalty
            loss_clash = self._clash_loss(pred_coords)

            # 5. Stacking reward
            loss_stack = self._stacking_loss(pred_coords)

            # 6. H-bond reward
            loss_hbond = self._hbond_loss(pred_coords, sequence)

            # 7. Suite torsion penalty
            loss_suite = self._suite_torsion_loss(pred_coords)

            total = (
                self.lambda_fape * loss_fape
                + self.lambda_distogram * loss_disto
                + self.lambda_bond * loss_bond
                + self.lambda_clash * loss_clash
                + self.lambda_stack * loss_stack
                + self.lambda_hbond * loss_hbond
                + self.lambda_suite * loss_suite
            )

            return total

        def _fape_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
            r"""Frame Aligned Point Error.

            :math:`\\text{FAPE} = \\frac{1}{L}\\sum_i \\sum_j
            \\min(|T_i^{-1} x_j^{\\text{pred}} - T_i^{-1} x_j^{\\text{true}}|, 10)`

            Simplified: using translations as frame origins (no rotation).

            Args:
                pred: Predicted coordinates, shape ``(L, 3)``.
                true: True coordinates, shape ``(L, 3)``.

            Returns:
                Scalar FAPE loss.
            """
            L = pred.shape[0]
            # Frame-relative coords: x_j - x_i for each frame i
            pred_rel = pred.unsqueeze(0) - pred.unsqueeze(1)  # (L, L, 3)
            true_rel = true.unsqueeze(0) - true.unsqueeze(1)  # (L, L, 3)
            err = (pred_rel - true_rel).norm(dim=-1)  # (L, L)
            err = err.clamp(max=10.0)
            return err.mean()

        def _distogram_loss(self, pred_logits: torch.Tensor,
                            true_bins: torch.Tensor) -> torch.Tensor:
            """Cross-entropy loss on distogram.

            Args:
                pred_logits: Shape ``(L, L, bins)``.
                true_bins: Shape ``(L, L)`` with integer bin indices.

            Returns:
                Scalar CE loss.
            """
            L = pred_logits.shape[0]
            return F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                true_bins.reshape(-1),
            )

        def _bond_geometry_loss(self, coords: torch.Tensor) -> torch.Tensor:
            r"""Bond length + angle geometry loss.

            :math:`L_{\\text{bond}} = \\frac{1}{L}\\sum_i (|x_{i+1}-x_i| - 5.9)^2
            + (\\theta_i - 2.62)^2`

            Uses 5.9Å C3'--C3' virtual bond from RNA PDB statistics.

            Args:
                coords: Shape ``(L, 3)``.

            Returns:
                Scalar bond geometry loss.
            """
            L = coords.shape[0]
            if L < 2:
                return torch.tensor(0.0, device=coords.device)

            # Bond length
            d = (coords[1:] - coords[:-1]).norm(dim=-1)  # (L-1,)
            bond_loss = ((d - 5.9) ** 2).mean()

            # Bond angle (for L >= 3)
            if L >= 3:
                v1 = coords[1:-1] - coords[:-2]
                v2 = coords[2:] - coords[1:-1]
                cos_angle = (F.cosine_similarity(v1, v2, dim=-1)).clamp(-1.0, 1.0)
                angles = torch.acos(cos_angle)
                angle_loss = ((angles - 2.62) ** 2).mean()
            else:
                angle_loss = torch.tensor(0.0, device=coords.device)

            return bond_loss + angle_loss

        def _clash_loss(self, coords: torch.Tensor) -> torch.Tensor:
            r"""Steric clash penalty.

            :math:`L_{\\text{clash}} = \\sum_{|i-j|>3} \\max(0, 3.0 - d_{ij})^2`

            Args:
                coords: Shape ``(L, 3)``.

            Returns:
                Scalar clash loss.
            """
            L = coords.shape[0]
            dists = torch.cdist(coords, coords)  # (L, L)
            idx = torch.arange(L, device=coords.device)
            sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > 3
            clash_vals = F.relu(3.0 - dists)
            clash_vals = clash_vals * sep_mask.float()
            return (clash_vals ** 2).sum() / max(L, 1)

        def _stacking_loss(self, coords: torch.Tensor) -> torch.Tensor:
            r"""Stacking reward for consecutive nucleotides.

            :math:`L_{\\text{stack}} = -\\frac{1}{L}\\sum_i
            \\exp\\bigl(-((d_{i,i+1} - 5.9)/1.0)^2\\bigr)`

            Args:
                coords: Shape ``(L, 3)``.

            Returns:
                Scalar stacking loss (negative = reward).
            """
            L = coords.shape[0]
            if L < 2:
                return torch.tensor(0.0, device=coords.device)

            d = (coords[1:] - coords[:-1]).norm(dim=-1)
            stack_reward = torch.exp(-((d - 5.9) / 1.0) ** 2)
            return -stack_reward.mean()

        def _hbond_loss(self, coords: torch.Tensor,
                        sequence: Optional[torch.Tensor] = None) -> torch.Tensor:
            r"""H-bond reward for Watson-Crick base pairs.

            Rewards C3'--C3' distance near :math:`10.4` Å for predicted WC pairs.

            Args:
                coords: Shape ``(L, 3)``.
                sequence: Encoded sequence, shape ``(L,)``.

            Returns:
                Scalar H-bond loss.
            """
            if sequence is None:
                return torch.tensor(0.0, device=coords.device)

            L = coords.shape[0]
            bp_dist_mean = self.bio.bp_dist_wc_mean  # 10.4 Å

            # WC complementarity mask: A-U (0-3) and G-C (2-1)
            seq_i = sequence.unsqueeze(1).expand(L, L)
            seq_j = sequence.unsqueeze(0).expand(L, L)
            wc_mask = (
                ((seq_i == 0) & (seq_j == 3)) |  # A-U
                ((seq_i == 3) & (seq_j == 0)) |  # U-A
                ((seq_i == 2) & (seq_j == 1)) |  # G-C
                ((seq_i == 1) & (seq_j == 2))     # C-G
            ).float()

            # Minimum sequence separation for base pairing
            idx = torch.arange(L, device=coords.device)
            sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= 4
            wc_mask = wc_mask * sep_mask.float()

            if wc_mask.sum() < 1:
                return torch.tensor(0.0, device=coords.device)

            dists = torch.cdist(coords, coords)
            hbond_penalty = wc_mask * ((dists - bp_dist_mean) ** 2)
            return hbond_penalty.sum() / (wc_mask.sum() + 1e-8)

        def _suite_torsion_loss(self, coords: torch.Tensor) -> torch.Tensor:
            r"""Suite torsion penalty from C3' pseudo-torsions.

            Computes pseudo-torsion angles from 3 consecutive C3' atoms,
            penalises distance to nearest of 10 Richardson suite clusters.

            :math:`L_{\\text{suite}} = \\frac{1}{L}\\sum_i
            \\min_c \\text{angular\\_dist}(\\theta_i, \\mu_c)`

            Args:
                coords: Shape ``(L, 3)``.

            Returns:
                Scalar suite torsion loss.
            """
            L = coords.shape[0]
            if L < 4:
                return torch.tensor(0.0, device=coords.device)

            # Compute pseudo-torsion from 4 consecutive C3' atoms
            v1 = coords[1:-2] - coords[:-3]
            v2 = coords[2:-1] - coords[1:-2]
            v3 = coords[3:] - coords[2:-1]

            n1 = torch.cross(v1, v2, dim=-1)
            n2 = torch.cross(v2, v3, dim=-1)

            n1_norm = n1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            n2_norm = n2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm

            cos_angle = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
            torsions = torch.acos(cos_angle)  # (L-3,)

            # Richardson suite cluster means (simplified: use alpha torsion angle)
            # Source: stage4_riemannian.py SUITE_CLUSTER_MEANS
            cluster_alphas = torch.tensor([
                math.radians(-68), math.radians(-68), math.radians(-70),
                math.radians(-58), math.radians(-68), math.radians(-66),
                math.radians(-65), math.radians(-66), math.radians(51),
                math.radians(-67),
            ], device=coords.device)

            # Angular distance to nearest cluster (wrapped)
            diffs = torsions.unsqueeze(-1) - cluster_alphas.unsqueeze(0)
            # Wrap to [-pi, pi]
            diffs = (diffs + math.pi) % (2 * math.pi) - math.pi
            min_dist = diffs.abs().min(dim=-1).values
            return min_dist.mean()
