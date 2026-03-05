"""Memory management for training on 310GB dataset with Colab T4 (15GB VRAM, 12GB RAM).

Monkey-patching + chunked loading + aggressive garbage collection.
Ensures the model never exceeds available VRAM by:
    - Loading CIF files in chunks of 80
    - Length bucketing to minimise padding waste
    - Random cropping for long sequences
    - ``gc.collect()`` + ``torch.cuda.empty_cache()`` after every batch
"""

from __future__ import annotations

import gc
import math
import os
import random
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch_not_available", msg="PyTorch not installed; memory manager limited")


class VRAMMonitor:
    """Monitors GPU VRAM usage.

    Wraps ``torch.cuda.memory_allocated()`` and ``torch.cuda.memory_reserved()``.
    """

    def __init__(self) -> None:
        self._cuda_available = _TORCH_AVAILABLE and torch.cuda.is_available()

    def used_gb(self) -> float:
        """Return currently allocated VRAM in GB."""
        if not self._cuda_available:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def free_gb(self, total_gb: float = 15.0) -> float:
        """Estimate free VRAM in GB.

        Args:
            total_gb: Total VRAM in GB.

        Returns:
            Estimated free VRAM.
        """
        return total_gb - self.used_gb()

    def log_status(self) -> None:
        """Log current VRAM usage."""
        if not self._cuda_available:
            logger.debug("vram_status", cuda_available=False)
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info("vram_status", allocated_gb=f"{allocated:.2f}", reserved_gb=f"{reserved:.2f}")

    def emergency_cleanup(self) -> None:
        """Empty CUDA cache, force GC, and log warning."""
        gc.collect()
        if self._cuda_available:
            torch.cuda.empty_cache()
        logger.warning("emergency_cleanup", msg="Freed VRAM via gc + cache empty")


if _TORCH_AVAILABLE:

    class ChunkedCIFDataset(IterableDataset):
        """Memory-safe CIF file dataset that loads files in chunks.

        Shuffles filenames → loads ``chunk_size`` CIFs → parses C3' coords →
        yields one by one → deletes chunk → ``gc.collect()``.

        Args:
            cif_dir: Directory containing .cif files.
            chunk_size: Number of CIF files to load per chunk.
            parse_fn: Function to parse a CIF file path into an RNAStructure.
        """

        def __init__(
            self,
            cif_dir: str,
            chunk_size: int = 80,
            parse_fn: Optional[Callable] = None,
            max_crop_len: int = 500,
            bucket_multiple: int = 32,
        ) -> None:
            super().__init__()
            self.cif_dir = cif_dir
            self.chunk_size = chunk_size
            self.max_crop_len = max_crop_len
            self.bucket_multiple = bucket_multiple

            if parse_fn is None:
                from .data_utils import parse_cif_c3_coords
                self.parse_fn = parse_cif_c3_coords
            else:
                self.parse_fn = parse_fn

            self._filenames: Optional[List[str]] = None

        def _get_filenames(self) -> List[str]:
            """Lazily discover CIF files."""
            if self._filenames is None:
                if os.path.isdir(self.cif_dir):
                    self._filenames = [
                        f for f in os.listdir(self.cif_dir)
                        if f.endswith(".cif")
                    ]
                else:
                    self._filenames = []
                    logger.warning("cif_dir_not_found", cif_dir=self.cif_dir)
            return self._filenames

        def __iter__(self) -> Iterator:
            """Yield parsed RNA structures in chunks."""
            filenames = self._get_filenames().copy()
            random.shuffle(filenames)

            for chunk_start in range(0, len(filenames), self.chunk_size):
                chunk_files = filenames[chunk_start:chunk_start + self.chunk_size]
                structures = []

                for fname in chunk_files:
                    path = os.path.join(self.cif_dir, fname)
                    try:
                        structure = self.parse_fn(path)
                        if structure is not None:
                            structures.append(structure)
                    except Exception as e:
                        logger.debug("cif_parse_skip", file=fname, error=str(e))

                # Length bucketing within chunk
                structures.sort(key=lambda s: len(s.sequence))

                for structure in structures:
                    seq = structure.sequence
                    coords = structure.coords_c3
                    L = len(seq)

                    # Random crop for long sequences
                    if L > self.max_crop_len:
                        start = random.randint(0, L - self.max_crop_len)
                        seq = seq[start:start + self.max_crop_len]
                        coords = coords[start:start + self.max_crop_len]
                        L = self.max_crop_len

                    # Pad to bucket multiple
                    pad_len = (self.bucket_multiple - L % self.bucket_multiple) % self.bucket_multiple
                    if pad_len > 0:
                        coords = np.pad(coords, ((0, pad_len), (0, 0)), mode="constant")
                        seq = seq + "A" * pad_len

                    yield {
                        "sequence": seq,
                        "coords": coords,
                        "length": L,
                        "pdb_id": structure.pdb_id,
                    }

                # Explicit cleanup
                del structures
                gc.collect()

    class MSAChunkLoader:
        """Load single MSA file on demand with subsampling.

        Subsamples to ``max_seqs`` rows (random without replacement).
        Returns one-hot encoded MSA matrix. Immediately deletes after returning.
        If MSA file missing: returns None (model handles sequence-only mode).

        Args:
            msa_dir: Directory containing MSA files.
            max_seqs: Maximum number of MSA sequences to keep.
            vocab_size: Vocabulary size for one-hot encoding.
        """

        def __init__(self, msa_dir: str, max_seqs: int = 128,
                     vocab_size: int = 5) -> None:
            self.msa_dir = msa_dir
            self.max_seqs = max_seqs
            self.vocab_size = vocab_size
            self._nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "-": 4}

        def load(self, seq_id: str) -> Optional[torch.Tensor]:
            """Load and subsample MSA for a sequence.

            Args:
                seq_id: Sequence identifier.

            Returns:
                One-hot MSA tensor ``(N, L, V)`` or None if unavailable.
            """
            # Try common MSA file patterns
            for ext in [".a3m", ".sto", ".fasta"]:
                path = os.path.join(self.msa_dir, seq_id + ext)
                if os.path.isfile(path):
                    return self._parse_msa(path)
            return None

        def _parse_msa(self, path: str) -> Optional[torch.Tensor]:
            """Parse MSA file and return one-hot encoded tensor."""
            try:
                sequences: List[str] = []
                current_seq = ""
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(">") or line.startswith("#"):
                            if current_seq:
                                sequences.append(current_seq)
                                current_seq = ""
                        elif line and not line.startswith("//"):
                            current_seq += line.upper().replace("T", "U")
                    if current_seq:
                        sequences.append(current_seq)

                if not sequences:
                    return None

                # Subsample
                if len(sequences) > self.max_seqs:
                    indices = random.sample(range(len(sequences)), self.max_seqs)
                    sequences = [sequences[i] for i in sorted(indices)]

                # One-hot encode
                max_len = max(len(s) for s in sequences)
                onehot = torch.zeros(len(sequences), max_len, self.vocab_size)
                for i, seq in enumerate(sequences):
                    for j, c in enumerate(seq):
                        idx = self._nuc_map.get(c, 4)
                        onehot[i, j, idx] = 1.0

                return onehot

            except Exception as e:
                logger.debug("msa_parse_error", path=path, error=str(e))
                return None
            finally:
                gc.collect()

    class CompetitionDataset(Dataset):
        """Dataset for competition CSV data (train_sequences.csv + train_labels.csv).

        Stores only sequence strings + coord arrays in memory (not raw CSV).
        Supports random crop + coordinate noise augmentation.

        Args:
            sequences_csv: Path to sequences CSV.
            labels_csv: Path to labels CSV.
            max_crop_len: Maximum crop length for long sequences.
            coord_noise_std: Standard deviation of coordinate noise augmentation.
            vocab_size: Vocabulary size for one-hot encoding.
        """

        def __init__(
            self,
            sequences_csv: str,
            labels_csv: str,
            max_crop_len: int = 500,
            coord_noise_std: float = 0.1,
            vocab_size: int = 5,
        ) -> None:
            super().__init__()
            from .data_utils import load_sequences_csv, load_labels_csv

            self.max_crop_len = max_crop_len
            self.coord_noise_std = coord_noise_std
            self.vocab_size = vocab_size
            self._nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

            # Load and store only essential data
            seq_records = load_sequences_csv(sequences_csv)
            labels_dict = load_labels_csv(labels_csv)

            self.entries: List[Tuple[str, str, np.ndarray]] = []
            for rec in seq_records:
                if rec.seq_id in labels_dict:
                    self.entries.append((
                        rec.seq_id,
                        rec.sequence,
                        labels_dict[rec.seq_id],
                    ))

            logger.info("competition_dataset_loaded", n_entries=len(self.entries))

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, idx: int) -> Dict:
            """Get a single training example.

            Returns:
                Dict with keys: seq_onehot, coords, length, seq_id.
            """
            seq_id, sequence, coords = self.entries[idx]
            L = len(sequence)

            # Random crop
            if L > self.max_crop_len:
                start = random.randint(0, L - self.max_crop_len)
                sequence = sequence[start:start + self.max_crop_len]
                coords = coords[start:start + self.max_crop_len]
                L = self.max_crop_len

            # One-hot encode
            onehot = torch.zeros(L, self.vocab_size)
            for i, c in enumerate(sequence):
                idx_c = self._nuc_map.get(c.upper(), 0)
                onehot[i, idx_c] = 1.0

            # Coordinate noise augmentation
            coords_tensor = torch.from_numpy(coords.copy()).float()
            if self.coord_noise_std > 0:
                coords_tensor += torch.randn_like(coords_tensor) * self.coord_noise_std

            return {
                "seq_onehot": onehot,
                "coords": coords_tensor,
                "length": L,
                "seq_id": seq_id,
            }

    class BucketBatchSampler(Sampler):
        """Groups sequences by length, yields indices from same bucket.

        Minimises padding waste by batching sequences of similar length.

        Args:
            lengths: List of sequence lengths.
            bucket_multiple: Bucket size multiple.
            batch_size: Batch size (number of sequences per batch).
            shuffle: Whether to shuffle within buckets.
        """

        def __init__(
            self,
            lengths: List[int],
            bucket_multiple: int = 32,
            batch_size: int = 1,
            shuffle: bool = True,
        ) -> None:
            self.lengths = lengths
            self.bucket_multiple = bucket_multiple
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self) -> Iterator[List[int]]:
            # Group indices by length bucket
            buckets: Dict[int, List[int]] = {}
            for i, length in enumerate(self.lengths):
                bucket_key = length // self.bucket_multiple
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(i)

            all_batches: List[List[int]] = []
            for indices in buckets.values():
                if self.shuffle:
                    random.shuffle(indices)
                for i in range(0, len(indices), self.batch_size):
                    batch = indices[i:i + self.batch_size]
                    all_batches.append(batch)

            if self.shuffle:
                random.shuffle(all_batches)

            yield from all_batches

        def __len__(self) -> int:
            return math.ceil(len(self.lengths) / self.batch_size)


def patch_dataloader_memory() -> None:
    """Monkey-patch DataLoader to insert GC after each batch.

    Patches ``DataLoader.__iter__`` to call ``gc.collect()`` and
    ``torch.cuda.empty_cache()`` after yielding each batch.
    Also patches ``DataLoader.__del__`` for cleanup.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("patch_skipped", msg="PyTorch not available")
        return

    _orig_iter = DataLoader.__iter__

    def _patched_iter(self: DataLoader) -> Iterator:
        for batch in _orig_iter(self):
            yield batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    DataLoader.__iter__ = _patched_iter  # type: ignore[assignment]

    def _patched_del(self: DataLoader) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    DataLoader.__del__ = _patched_del  # type: ignore[assignment]
    logger.info("dataloader_memory_patched")
