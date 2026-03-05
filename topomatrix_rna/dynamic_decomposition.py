"""Adaptive domain decomposition — threshold computed at runtime from VRAM + sequence features.

Dynamically computes decomposition thresholds based on available VRAM,
sequence composition (GC content), and predicted contact density.

Mathematical Basis:
    :math:`L_{\\max} = \\lfloor\\sqrt{\\frac{(V_{\\text{total}} -
    V_{\\text{used}} - V_{\\text{overhead}}) \\times 10^9}
    {d_z \\times 2 \\times S}}\\rfloor`

    where :math:`V` is VRAM in GB, :math:`d_z` is pair dim, :math:`S`
    is safety factor.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import structlog

from .config import DomainConfig, MemoryConfig, PipelineConfig

logger = structlog.get_logger(__name__)


class AdaptiveDomainDecomposer:
    """Adaptive domain decomposition with runtime VRAM-based thresholds.

    Args:
        config: PipelineConfig or a compatible object with ``domain``,
            ``memory``, and ``physics_net`` attributes.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        if config is None:
            config = PipelineConfig()
        self.config = config
        self.domain_config = config.domain
        self.memory_config = config.memory

    def compute_threshold(
        self,
        sequence: str,
        vram_gb: float = 15.0,
        current_vram_used_gb: float = 0.0,
    ) -> int:
        r"""Compute maximum processable sequence length from VRAM budget.

        :math:`L_{\\max} = \\lfloor\\sqrt{\\frac{(V - V_{\\text{used}} -
        V_{\\text{overhead}}) \\times 10^9}{d_z \\times 2 \\times S}}\\rfloor`

        Then adjusted for GC content and contact density heuristics.

        Args:
            sequence: RNA sequence string.
            vram_gb: Total available VRAM in GB.
            current_vram_used_gb: Currently used VRAM in GB.

        Returns:
            Maximum processable sequence length (clamped to [100, 2000]).
        """
        mem_cfg = self.memory_config
        pair_dim = self.config.physics_net.base_pair_dim
        safety = mem_cfg.safety_factor

        free_bytes = (vram_gb - current_vram_used_gb - mem_cfg.overhead_gb) * 1e9
        free_bytes = max(free_bytes, 1e6)  # Floor at 1MB

        max_L = int(math.floor(math.sqrt(free_bytes / (pair_dim * 2 * safety))))

        # GC content adjustment
        gc_count = sum(1 for c in sequence.upper() if c in ("G", "C"))
        gc_content = gc_count / max(len(sequence), 1)
        if gc_content > 0.6:
            max_L = int(max_L * 1.1)  # +10% for GC-rich sequences

        # Contact density heuristic (rough estimate from sequence composition)
        L = len(sequence)
        complement_count = 0
        for i in range(L):
            for j in range(i + 4, min(i + 30, L)):
                ci, cj = sequence[i].upper(), sequence[j].upper()
                if (ci, cj) in (("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")):
                    complement_count += 1
        contact_density = complement_count / max(L * 26, 1)
        if contact_density > 0.15:
            max_L = int(max_L * 0.8)  # -20% for high contact density

        # Clamp
        max_L = max(100, min(max_L, 2000))

        logger.debug(
            "adaptive_threshold",
            max_L=max_L, gc_content=f"{gc_content:.2f}",
            contact_density=f"{contact_density:.3f}",
            vram_free_gb=f"{vram_gb - current_vram_used_gb:.1f}",
        )

        return max_L

    def should_decompose(self, L: int, sequence: str,
                         vram_gb: float = 15.0,
                         current_vram_used_gb: float = 0.0) -> bool:
        """Check if a sequence should be decomposed.

        Args:
            L: Sequence length.
            sequence: RNA sequence string.
            vram_gb: Total VRAM.
            current_vram_used_gb: Currently used VRAM.

        Returns:
            True if decomposition is needed.
        """
        threshold = self.compute_threshold(sequence, vram_gb, current_vram_used_gb)
        return L > threshold

    def decompose(
        self,
        sequence: str,
        contact_map: np.ndarray,
        vram_gb: float = 15.0,
        current_vram_used_gb: float = 0.0,
    ) -> List[Tuple[int, int]]:
        """Decompose sequence into domains using existing SpectralDomainDecomposer.

        Uses the dynamic threshold computed from VRAM budget.

        Args:
            sequence: RNA sequence.
            contact_map: Contact probability matrix, shape ``(L, L)``.
            vram_gb: Total VRAM.
            current_vram_used_gb: Currently used VRAM.

        Returns:
            List of (start, end) domain boundaries.
        """
        from .stage7_domain import SpectralDomainDecomposer

        L = len(sequence)
        threshold = self.compute_threshold(sequence, vram_gb, current_vram_used_gb)

        # Create domain config with dynamic threshold
        domain_cfg = DomainConfig(
            use_threshold_length=threshold,
            min_domain_size=self.domain_config.min_domain_size,
            max_domain_size=self.domain_config.max_domain_size,
            se3_lr=self.domain_config.se3_lr,
            se3_steps=self.domain_config.se3_steps,
        )

        decomposer = SpectralDomainDecomposer(domain_cfg)
        return decomposer.decompose(contact_map, L)

    def assemble(
        self,
        domain_coords: List[np.ndarray],
        domain_boundaries: List[Tuple[int, int]],
        contact_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assemble domain coordinates into full structure.

        Args:
            domain_coords: List of per-domain coordinate arrays.
            domain_boundaries: List of (start, end) boundaries.
            contact_map: Optional contact map for refinement.

        Returns:
            Assembled coordinates, shape ``(L, 3)``.
        """
        from .stage7_domain import SE3DomainAssembler

        assembler = SE3DomainAssembler(self.domain_config)
        if contact_map is None:
            L = domain_boundaries[-1][1] if domain_boundaries else 0
            contact_map = np.zeros((L, L))

        return assembler.assemble(domain_coords, domain_boundaries, contact_map)
