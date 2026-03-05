"""ONNX inference wrapper with graceful fallback.

Provides ``PhysicsNetPredictor`` that tries ONNX inference first,
falling back to A-form helix generation if the ONNX model is missing.

Graceful degradation:
    - GPU unavailable → CPU
    - ONNX missing → A-form helix fallback
    - MSA missing → sequence-only mode
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import structlog

from .config import PipelineConfig, RNABiologyConstants

logger = structlog.get_logger(__name__)


def generate_aform_helix(sequence: str, biology: Optional[RNABiologyConstants] = None
                         ) -> np.ndarray:
    r"""Generate A-form helix coordinates for an RNA sequence.

    Uses canonical A-form RNA helix parameters:
    :math:`\\text{rise} = 2.81` Å, :math:`\\text{twist} = 32.7°`,
    :math:`\\text{radius} = 9.0` Å.

    Always produces valid output regardless of sequence.

    Args:
        sequence: RNA sequence string.
        biology: Optional biology constants.

    Returns:
        C3' coordinates, shape ``(L, 3)``.
    """
    if biology is None:
        biology = RNABiologyConstants()

    L = len(sequence)
    coords = np.zeros((L, 3), dtype=np.float64)

    rise = biology.helix_rise_a_form     # 2.81 Å
    twist = biology.helix_twist_a_form   # 32.7 degrees
    radius = 9.0                          # Å

    twist_rad = math.radians(twist)

    for i in range(L):
        angle = i * twist_rad
        coords[i, 0] = radius * math.cos(angle)
        coords[i, 1] = radius * math.sin(angle)
        coords[i, 2] = i * rise

    return coords


class PhysicsNetPredictor:
    """ONNX inference wrapper with graceful fallback to A-form helix.

    Args:
        onnx_path: Path to ONNX model file.
        config: Pipeline configuration.
    """

    def __init__(self, onnx_path: str,
                 config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.onnx_path = onnx_path
        self.fallback = False
        self.session = None
        self._nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path)
            logger.info("onnx_loaded", path=onnx_path)
        except Exception as e:
            self.fallback = True
            logger.warning("onnx_load_failed", path=onnx_path, error=str(e),
                           msg="Using A-form helix fallback")

    def predict(self, sequence: str) -> np.ndarray:
        """Predict 3D coordinates for an RNA sequence.

        If ONNX model is unavailable, generates A-form helix coordinates.
        For long sequences exceeding dynamic threshold, uses domain
        decomposition.

        Args:
            sequence: RNA sequence string (ACGU).

        Returns:
            C3' coordinates, shape ``(L, 3)``.
        """
        L = len(sequence)

        # Check if decomposition needed
        from .dynamic_decomposition import AdaptiveDomainDecomposer
        decomposer = AdaptiveDomainDecomposer(self.config)

        try:
            from .memory_manager import VRAMMonitor
            vram_used = VRAMMonitor().used_gb()
        except Exception:
            vram_used = 0.0

        if decomposer.should_decompose(L, sequence,
                                       vram_gb=self.config.memory.vram_gb,
                                       current_vram_used_gb=vram_used):
            return self._predict_decomposed(sequence, decomposer)

        return self._predict_single(sequence)

    def predict_with_confidence(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict coordinates with per-residue confidence (pLDDT).

        Args:
            sequence: RNA sequence string.

        Returns:
            Tuple of (coords ``(L, 3)``, plddt ``(L,)``).
        """
        coords = self.predict(sequence)
        L = len(sequence)

        if self.fallback:
            # Fallback: uniform confidence of 0.5
            plddt = np.full(L, 0.5, dtype=np.float64)
        else:
            try:
                onehot = self._encode_sequence(sequence)
                outputs = self.session.run(None, {"seq_onehot": onehot})
                plddt = outputs[2].flatten()[:L]
            except Exception:
                plddt = np.full(L, 0.5, dtype=np.float64)

        return coords, plddt

    def _predict_single(self, sequence: str) -> np.ndarray:
        """Predict coordinates for a single sequence.

        Args:
            sequence: RNA sequence.

        Returns:
            Coordinates ``(L, 3)``.
        """
        if self.fallback:
            return generate_aform_helix(sequence, self.config.biology)

        try:
            onehot = self._encode_sequence(sequence)
            outputs = self.session.run(None, {"seq_onehot": onehot})
            coords = outputs[0]  # (L, 3)
            return coords.astype(np.float64)
        except Exception as e:
            logger.warning("onnx_inference_error", error=str(e),
                           msg="Falling back to A-form helix")
            return generate_aform_helix(sequence, self.config.biology)

    def _predict_decomposed(self, sequence: str,
                            decomposer: object) -> np.ndarray:
        """Predict using domain decomposition for long sequences.

        Args:
            sequence: RNA sequence.
            decomposer: AdaptiveDomainDecomposer instance.

        Returns:
            Assembled coordinates ``(L, 3)``.
        """
        from .stage1_contact_map import ContactMapPredictor
        from .dynamic_decomposition import AdaptiveDomainDecomposer

        L = len(sequence)

        # Generate contact map for decomposition
        contact_predictor = ContactMapPredictor(self.config.contact)
        contact_map = contact_predictor.predict(sequence, return_sparse=False)

        # Decompose
        assert isinstance(decomposer, AdaptiveDomainDecomposer)
        domains = decomposer.decompose(sequence, contact_map)

        # Predict each domain
        domain_coords = []
        for start, end in domains:
            sub_seq = sequence[start:end]
            sub_coords = self._predict_single(sub_seq)
            domain_coords.append(sub_coords)

        # Assemble
        return decomposer.assemble(domain_coords, domains, contact_map)

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """One-hot encode a sequence for ONNX inference.

        Args:
            sequence: RNA sequence string.

        Returns:
            One-hot array, shape ``(L, 5)``.
        """
        L = len(sequence)
        vocab_size = self.config.physics_net.vocab_size
        onehot = np.zeros((L, vocab_size), dtype=np.float32)
        for i, c in enumerate(sequence.upper()):
            idx = self._nuc_map.get(c, 0)
            onehot[i, idx] = 1.0
        return onehot
