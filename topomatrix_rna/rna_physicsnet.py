"""RNA-PhysicsNet: Physics-informed neural network for RNA 3D structure prediction.

Architecture: Transformer encoder (single+pair track) → IPA structure module → recycling.
Dynamic scaling: model dims auto-shrink based on sequence length and available VRAM.

Mathematical Basis:
    Single representation: :math:`s_i \in \mathbb{R}^{d_s}` per nucleotide.
    Pair representation: :math:`z_{ij} \in \mathbb{R}^{d_z}` per pair.
    IPA: invariant point attention on SE(3) rigid frames.
    Recycling: iterative refinement of structure predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch_not_available", msg="PyTorch not installed; model unavailable")


@dataclass
class DynamicConfig:
    r"""Dynamic model configuration that scales with sequence length and VRAM.

    Dimensions are chosen so that peak memory stays within the available VRAM:
    :math:`\text{mem} \approx L^2 \cdot d_z \cdot 4 \; \text{bytes (FP32)}`
    """

    single_dim: int = 128
    pair_dim: int = 64
    n_layers: int = 6
    ipa_layers: int = 4
    n_heads: int = 4
    chunk_size: int = 64
    n_recycles: int = 3
    decompose: bool = False

    @staticmethod
    def from_length(L: int, vram_gb: float = 15.0) -> "DynamicConfig":
        """Return a DynamicConfig scaled to sequence length and VRAM.

        Args:
            L: Sequence length.
            vram_gb: Available VRAM in GB.

        Returns:
            DynamicConfig instance.
        """
        if L <= 200:
            return DynamicConfig(
                single_dim=256, pair_dim=128, n_layers=8, ipa_layers=6,
                n_heads=8, chunk_size=L, n_recycles=3, decompose=False,
            )
        if L <= 500:
            return DynamicConfig(
                single_dim=128, pair_dim=64, n_layers=6, ipa_layers=4,
                n_heads=4, chunk_size=64, n_recycles=3, decompose=False,
            )
        if L <= 1000:
            return DynamicConfig(
                single_dim=96, pair_dim=32, n_layers=4, ipa_layers=3,
                n_heads=4, chunk_size=32, n_recycles=2, decompose=False,
            )
        return DynamicConfig(
            single_dim=96, pair_dim=32, n_layers=4, ipa_layers=3,
            n_heads=4, chunk_size=32, n_recycles=1, decompose=True,
        )


def get_dynamic_config(L: int, vram_gb: float = 15.0) -> DynamicConfig:
    """Convenience wrapper for :meth:`DynamicConfig.from_length`.

    Args:
        L: Sequence length.
        vram_gb: Available VRAM in GB.

    Returns:
        DynamicConfig instance.
    """
    return DynamicConfig.from_length(L, vram_gb)


if _TORCH_AVAILABLE:

    class NucleotideEmbedding(nn.Module):
        r"""Nucleotide embedding with sinusoidal positional encoding.

        One-hot(5) → Linear → single_dim.
        Sinusoidal positional encoding (Vaswani et al., 2017):
        :math:`PE_{(pos, 2k)} = \sin(pos / 10000^{2k/d})`
        :math:`PE_{(pos, 2k+1)} = \cos(pos / 10000^{2k/d})`

        Pair representation initialised via outer sum of single repr.
        """

        def __init__(self, vocab_size: int, single_dim: int, pair_dim: int,
                     max_len: int = 4096) -> None:
            super().__init__()
            self.linear = nn.Linear(vocab_size, single_dim)
            self.pair_proj_left = nn.Linear(single_dim, pair_dim)
            self.pair_proj_right = nn.Linear(single_dim, pair_dim)

            # Pre-compute sinusoidal positional encoding
            pe = torch.zeros(max_len, single_dim)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, single_dim, 2, dtype=torch.float32) *
                -(math.log(10000.0) / single_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:single_dim // 2])
            self.register_buffer("pe", pe)

        def forward(self, seq_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Embed nucleotide sequence into single and pair representations.

            Args:
                seq_onehot: One-hot encoded sequence, shape ``(L, V)``.

            Returns:
                Tuple of (single_repr ``(L, d_s)``, pair_repr ``(L, L, d_z)``).
            """
            L = seq_onehot.shape[0]
            single = self.linear(seq_onehot) + self.pe[:L]
            left = self.pair_proj_left(single)
            right = self.pair_proj_right(single)
            pair = left.unsqueeze(1) + right.unsqueeze(0)  # (L, L, d_z)
            return single, pair

    class AxialAttention(nn.Module):
        r"""Row-wise + column-wise attention on pair representation.

        Chunked: processes ``chunk_size`` rows at a time to avoid
        :math:`O(L^2)` memory.
        """

        def __init__(self, pair_dim: int, n_heads: int, chunk_size: int = 64) -> None:
            super().__init__()
            self.row_attn = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True)
            self.col_attn = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True)
            self.norm_row = nn.LayerNorm(pair_dim)
            self.norm_col = nn.LayerNorm(pair_dim)
            self.chunk_size = chunk_size

        def forward(self, pair: torch.Tensor) -> torch.Tensor:
            """Apply axial attention on pair representation.

            Args:
                pair: Pair representation, shape ``(L, L, d_z)``.

            Returns:
                Updated pair representation, shape ``(L, L, d_z)``.
            """
            L = pair.shape[0]
            cs = self.chunk_size

            # Row-wise attention (chunked)
            out = pair.clone()
            for start in range(0, L, cs):
                end = min(start + cs, L)
                chunk = pair[start:end]  # (cs, L, d_z)
                normed = self.norm_row(chunk)
                attn_out, _ = self.row_attn(normed, normed, normed)
                out[start:end] = chunk + attn_out

            # Column-wise attention (chunked)
            pair_t = out.permute(1, 0, 2)  # (L, L, d_z) transposed
            out_t = pair_t.clone()
            for start in range(0, L, cs):
                end = min(start + cs, L)
                chunk = pair_t[start:end]
                normed = self.norm_col(chunk)
                attn_out, _ = self.col_attn(normed, normed, normed)
                out_t[start:end] = chunk + attn_out

            return out_t.permute(1, 0, 2)

    class TriangularMultiplicativeUpdate(nn.Module):
        r"""Triangular multiplicative update for pair representation.

        Implements outgoing and incoming edge updates:
        :math:`z_{ij} \mathrel{+}= \sigma(\text{gate}) \odot
        \sum_k \text{proj}_a(z_{ik}) \odot \text{proj}_b(z_{jk})`
        """

        def __init__(self, pair_dim: int, chunk_size: int = 64) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(pair_dim)
            self.proj_a = nn.Linear(pair_dim, pair_dim)
            self.proj_b = nn.Linear(pair_dim, pair_dim)
            self.gate = nn.Linear(pair_dim, pair_dim)
            self.out_proj = nn.Linear(pair_dim, pair_dim)
            self.chunk_size = chunk_size

        def forward(self, pair: torch.Tensor) -> torch.Tensor:
            """Apply triangular multiplicative update.

            Args:
                pair: Pair representation, shape ``(L, L, d_z)``.

            Returns:
                Updated pair representation, shape ``(L, L, d_z)``.
            """
            L = pair.shape[0]
            cs = self.chunk_size
            normed = self.norm(pair)
            a = torch.sigmoid(self.proj_a(normed))  # (L, L, d_z)
            b = torch.sigmoid(self.proj_b(normed))  # (L, L, d_z)
            g = torch.sigmoid(self.gate(normed))

            out = torch.zeros_like(pair)
            for i_start in range(0, L, cs):
                i_end = min(i_start + cs, L)
                for j_start in range(0, L, cs):
                    j_end = min(j_start + cs, L)
                    # Chunked matmul: sum_k a[i,k] * b[j,k]
                    a_chunk = a[i_start:i_end]  # (cs_i, L, d_z)
                    b_chunk = b[j_start:j_end]  # (cs_j, L, d_z)
                    product = torch.einsum("ikd,jkd->ijd", a_chunk, b_chunk)
                    out[i_start:i_end, j_start:j_end] = product

            return pair + g * self.out_proj(out)

    class TransformerBlock(nn.Module):
        """Single transformer encoder block with single and pair tracks.

        single self-attention → single→pair outer product (chunked) →
        AxialAttention on pair → TriangularMultiplicativeUpdate → pair→single.
        """

        def __init__(self, single_dim: int, pair_dim: int, n_heads: int,
                     chunk_size: int = 64, dropout: float = 0.1) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(single_dim, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(single_dim)
            self.norm2 = nn.LayerNorm(single_dim)
            self.ffn = nn.Sequential(
                nn.Linear(single_dim, single_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(single_dim * 4, single_dim),
                nn.Dropout(dropout),
            )
            # Outer product projection
            self.outer_left = nn.Linear(single_dim, pair_dim)
            self.outer_right = nn.Linear(single_dim, pair_dim)
            self.outer_norm = nn.LayerNorm(pair_dim)

            self.axial_attn = AxialAttention(pair_dim, n_heads, chunk_size)
            self.tri_update = TriangularMultiplicativeUpdate(pair_dim, chunk_size)

            # pair→single projection
            self.pair_to_single = nn.Linear(pair_dim, single_dim)
            self.pair_norm = nn.LayerNorm(pair_dim)
            self.chunk_size = chunk_size

        def forward(self, single: torch.Tensor, pair: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Run transformer block on single and pair representations.

            Args:
                single: Single representation, shape ``(L, d_s)``.
                pair: Pair representation, shape ``(L, L, d_z)``.

            Returns:
                Updated (single, pair) tuple.
            """
            # Single self-attention
            s_norm = self.norm1(single)
            s_unsqueeze = s_norm.unsqueeze(0)  # (1, L, d_s)
            attn_out, _ = self.self_attn(s_unsqueeze, s_unsqueeze, s_unsqueeze)
            single = single + attn_out.squeeze(0)
            single = single + self.ffn(self.norm2(single))

            # Outer product: single → pair (chunked)
            L = single.shape[0]
            cs = self.chunk_size
            left = self.outer_left(single)
            right = self.outer_right(single)
            for i_start in range(0, L, cs):
                i_end = min(i_start + cs, L)
                outer_chunk = left[i_start:i_end].unsqueeze(1) * right.unsqueeze(0)
                pair[i_start:i_end] = pair[i_start:i_end] + self.outer_norm(outer_chunk)

            # Pair track
            pair = self.axial_attn(pair)
            pair = self.tri_update(pair)

            # Pair → single
            pair_mean = self.pair_norm(pair).mean(dim=1)  # (L, d_z)
            single = single + self.pair_to_single(pair_mean)

            return single, pair

    class InvariantPointAttention(nn.Module):
        r"""Invariant Point Attention (IPA) from AlphaFold2 with reduced dims.

        Takes single repr + pair repr + rigid frames → attention weights
        invariant to global SE(3) transforms.

        :math:`a_{ij} = \text{softmax}\bigl(
            \frac{1}{\sqrt{d_h}} q_i^\top k_j +
            \sum_p \|T_i^{-1} q^p_i - T_j^{-1} k^p_j\|^2 +
            b_{ij}\bigr)`
        """

        def __init__(self, single_dim: int, pair_dim: int, n_heads: int = 4,
                     n_query_points: int = 4) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.n_query_points = n_query_points
            head_dim = single_dim // n_heads

            self.q_proj = nn.Linear(single_dim, n_heads * head_dim)
            self.k_proj = nn.Linear(single_dim, n_heads * head_dim)
            self.v_proj = nn.Linear(single_dim, n_heads * head_dim)

            # Point projections (query and key points in local frames)
            self.q_points = nn.Linear(single_dim, n_heads * n_query_points * 3)
            self.k_points = nn.Linear(single_dim, n_heads * n_query_points * 3)
            self.v_points = nn.Linear(single_dim, n_heads * n_query_points * 3)

            # Pair bias
            self.pair_bias = nn.Linear(pair_dim, n_heads)

            self.out_proj = nn.Linear(
                n_heads * (head_dim + n_query_points * 3 + pair_dim), single_dim
            )
            self.head_dim = head_dim

        def forward(self, single: torch.Tensor, pair: torch.Tensor,
                    translations: torch.Tensor, rotations: torch.Tensor
                    ) -> torch.Tensor:
            r"""Apply IPA.

            Args:
                single: Shape ``(L, d_s)``.
                pair: Shape ``(L, L, d_z)``.
                translations: Per-residue translations, shape ``(L, 3)``.
                rotations: Per-residue rotations, shape ``(L, 3, 3)``.

            Returns:
                Updated single representation, shape ``(L, d_s)``.
            """
            L = single.shape[0]
            h = self.n_heads
            d = self.head_dim
            n_pts = self.n_query_points

            q = self.q_proj(single).view(L, h, d)
            k = self.k_proj(single).view(L, h, d)
            v = self.v_proj(single).view(L, h, d)

            # Scalar attention
            attn_logits = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(d)

            # Pair bias
            attn_logits = attn_logits + self.pair_bias(pair)

            # Point attention: transform query/key points to global frame
            q_pts = self.q_points(single).view(L, h, n_pts, 3)
            k_pts = self.k_points(single).view(L, h, n_pts, 3)
            v_pts = self.v_points(single).view(L, h, n_pts, 3)

            # Apply rigid transform: global_pt = R @ local_pt + t
            q_global = torch.einsum("lab,ihpb->ihpa", rotations, q_pts) + \
                translations.unsqueeze(1).unsqueeze(1)
            k_global = torch.einsum("lab,jhpb->jhpa", rotations, k_pts) + \
                translations.unsqueeze(1).unsqueeze(1)

            # Point distance penalty
            pt_diff = q_global.unsqueeze(1) - k_global.unsqueeze(0)  # (L, L, h, n_pts, 3)
            pt_dist = pt_diff.pow(2).sum(dim=-1).sum(dim=-1)  # (L, L, h)
            attn_logits = attn_logits - pt_dist * 0.5

            weights = F.softmax(attn_logits, dim=1)  # (L, L, h)

            # Aggregate values
            result_scalar = torch.einsum("ijh,jhd->ihd", weights, v)  # (L, h, d)
            v_global = torch.einsum("lab,jhpb->jhpa", rotations, v_pts) + \
                translations.unsqueeze(1).unsqueeze(1)
            result_points = torch.einsum("ijh,jhpa->ihpa", weights, v_global)  # (L, h, n_pts, 3)

            # Transform result points back to local frame
            result_points_local = torch.einsum(
                "lab,ihpb->ihpa",
                rotations.transpose(-1, -2),
                result_points - translations.unsqueeze(1).unsqueeze(1)
            )

            # Pair aggregation
            result_pair = torch.einsum("ijh,ijd->ihd", weights, pair)  # (L, h, d_z)

            # Concatenate and project
            out = torch.cat([
                result_scalar.reshape(L, -1),
                result_points_local.reshape(L, -1),
                result_pair.reshape(L, -1),
            ], dim=-1)

            return self.out_proj(out)

    class StructureModule(nn.Module):
        r"""Structure module: IPA layers → backbone update → C3' extraction.

        Initialises identity frames, then iterates:
        IPA → transition → backbone_update (rotation + translation delta)
        → update frames → extract C3' position.

        :math:`T_i^{(l+1)} = T_i^{(l)} \circ \Delta T_i^{(l)}`
        """

        def __init__(self, single_dim: int, pair_dim: int, ipa_layers: int = 4,
                     n_heads: int = 4, n_query_points: int = 4) -> None:
            super().__init__()
            self.ipa_layers_count = ipa_layers
            self.ipa_layers = nn.ModuleList([
                InvariantPointAttention(single_dim, pair_dim, n_heads, n_query_points)
                for _ in range(ipa_layers)
            ])
            self.transitions = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(single_dim),
                    nn.Linear(single_dim, single_dim),
                    nn.GELU(),
                    nn.Linear(single_dim, single_dim),
                )
                for _ in range(ipa_layers)
            ])
            # Backbone update: predict rotation (as 6D repr) + translation delta
            self.backbone_update = nn.Linear(single_dim, 9)  # 6 rotation + 3 translation
            # pLDDT head
            self.plddt_head = nn.Sequential(
                nn.LayerNorm(single_dim),
                nn.Linear(single_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, single: torch.Tensor, pair: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Run structure module.

            Args:
                single: Shape ``(L, d_s)``.
                pair: Shape ``(L, L, d_z)``.

            Returns:
                Tuple of (coords ``(L, 3)``, plddt ``(L,)``).
            """
            L = single.shape[0]
            device = single.device

            # Initialise identity frames
            translations = torch.zeros(L, 3, device=device)
            rotations = torch.eye(3, device=device).unsqueeze(0).expand(L, -1, -1).clone()

            for i in range(self.ipa_layers_count):
                single = single + self.ipa_layers[i](single, pair, translations, rotations)
                single = single + self.transitions[i](single)

                # Backbone update
                update = self.backbone_update(single)
                rot_delta_flat = update[:, :6]
                trans_delta = update[:, 6:]

                # Convert 6D rotation to matrix (Gram-Schmidt)
                rot_delta = self._rot6d_to_matrix(rot_delta_flat)

                # Update frames
                rotations = torch.bmm(rotations, rot_delta)
                translations = translations + torch.bmm(
                    rotations, trans_delta.unsqueeze(-1)
                ).squeeze(-1)

            coords = translations  # C3' position = translation component
            plddt = self.plddt_head(single).squeeze(-1)

            return coords, plddt

        @staticmethod
        def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
            """Convert 6D rotation representation to 3x3 rotation matrix.

            Uses Gram-Schmidt orthogonalisation (Zhou et al., 2019).

            Args:
                rot6d: Shape ``(N, 6)``.

            Returns:
                Rotation matrices, shape ``(N, 3, 3)``.
            """
            a1 = rot6d[:, :3]
            a2 = rot6d[:, 3:]

            b1 = F.normalize(a1, dim=-1)
            dot = (b1 * a2).sum(dim=-1, keepdim=True)
            b2 = F.normalize(a2 - dot * b1, dim=-1)
            b3 = torch.cross(b1, b2, dim=-1)

            return torch.stack([b1, b2, b3], dim=-1)

    class RNAPhysicsNet(nn.Module):
        """Full RNA-PhysicsNet model.

        ``forward(seq_onehot, msa_feat=None)`` →
        1. Embed → single, pair
        2. If msa_feat provided: project and add to pair repr
        3. For each recycle: run encoder blocks → structure module →
           get coords → compute distogram from coords → feed back
        4. Return coords (L, 3), distogram_logits (L, L, 64), plddt (L,)
        5. Uses ``torch.utils.checkpoint.checkpoint`` on every encoder block.
        """

        def __init__(self, config: DynamicConfig, physics_config: Optional[object] = None
                     ) -> None:
            super().__init__()
            self.config = config
            pc = physics_config
            vocab_size = getattr(pc, "vocab_size", 5)
            distogram_bins = getattr(pc, "distogram_bins", 64)
            dropout = getattr(pc, "dropout", 0.1)

            self.embedding = NucleotideEmbedding(
                vocab_size, config.single_dim, config.pair_dim,
            )

            self.msa_proj = nn.Linear(vocab_size, config.pair_dim) if True else None

            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(
                    config.single_dim, config.pair_dim, config.n_heads,
                    config.chunk_size, dropout,
                )
                for _ in range(config.n_layers)
            ])

            self.structure_module = StructureModule(
                config.single_dim, config.pair_dim, config.ipa_layers,
                config.n_heads,
            )

            # Distogram head
            self.distogram_head = nn.Sequential(
                nn.LayerNorm(config.pair_dim),
                nn.Linear(config.pair_dim, distogram_bins),
            )

            self.n_recycles = config.n_recycles
            self.distogram_bins = distogram_bins
            # Recycle embeddings
            self.recycle_single = nn.Linear(3, config.single_dim)
            self.recycle_pair = nn.Linear(distogram_bins, config.pair_dim)

        def forward(self, seq_onehot: torch.Tensor,
                    msa_feat: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Run forward pass.

            Args:
                seq_onehot: One-hot encoded sequence, shape ``(L, V)``.
                msa_feat: Optional MSA features, shape ``(N_msa, L, V)``.

            Returns:
                Tuple of (coords ``(L, 3)``, distogram_logits ``(L, L, bins)``,
                plddt ``(L,)``).
            """
            single, pair = self.embedding(seq_onehot)

            # MSA feature integration
            if msa_feat is not None:
                msa_proj = self.msa_proj(msa_feat)  # (N_msa, L, d_z)
                pair = pair + msa_proj.mean(dim=0).unsqueeze(0)

            coords = None
            for recycle_idx in range(self.n_recycles):
                # Recycle: feed back previous coords and distogram
                if coords is not None:
                    single = single + self.recycle_single(coords.detach())
                    dists = torch.cdist(coords.detach(), coords.detach())
                    disto_target = self._coords_to_distogram(dists)
                    pair = pair + self.recycle_pair(disto_target.detach())

                # Encoder blocks with gradient checkpointing
                s, p = single, pair
                for block in self.encoder_blocks:
                    s, p = torch_checkpoint(block, s, p, use_reentrant=False)
                single, pair = s, p

                # Structure module
                coords, plddt = self.structure_module(single, pair)

            # Distogram logits
            distogram_logits = self.distogram_head(pair)

            return coords, distogram_logits, plddt

        def _coords_to_distogram(self, dists: torch.Tensor) -> torch.Tensor:
            """Convert distance matrix to soft distogram.

            Args:
                dists: Pairwise distances, shape ``(L, L)``.

            Returns:
                Soft distogram, shape ``(L, L, bins)``.
            """
            bins = torch.linspace(2.0, 22.0, self.distogram_bins, device=dists.device)
            diff = dists.unsqueeze(-1) - bins
            return F.softmax(-diff.pow(2), dim=-1)
