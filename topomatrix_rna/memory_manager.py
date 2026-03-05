"""Memory management for training on 310GB dataset with Colab T4 (15GB VRAM, 12GB RAM).

Monkey-patching + chunked loading + aggressive garbage collection.
Ensures the model never exceeds available VRAM by:
    - Loading CIF files in chunks of 80
    - Length bucketing to minimise padding waste
    - Random cropping for long sequences
    - ``gc.collect()`` + ``torch.cuda.empty_cache()`` after every batch

Streaming classes (DiskCacheManager, StreamingCIFDataset, StreamingMSALoader) allow
on-demand download of individual files from Kaggle so that disk usage stays bounded
(configurable ``max_disk_cache_gb``).  The lazy-loading ``CompetitionDataset`` avoids
holding all label coordinates in RAM at once.
"""

from __future__ import annotations

import csv
import gc
import math
import os
import random
import shutil
import subprocess
import time
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

        When ``lazy_labels=True`` (default ``False`` for backwards compatibility), label
        coordinates are **not** loaded into RAM at init time.  Instead, each ``__getitem__``
        call reads only the required rows from the labels CSV using a chunked scan.  This
        keeps peak RAM usage proportional to ``labels_chunk_size`` rather than the full
        labels file.

        When ``lazy_labels=False`` (original behaviour), all labels are loaded at init.

        Args:
            sequences_csv: Path to sequences CSV.
            labels_csv: Path to labels CSV.
            max_crop_len: Maximum crop length for long sequences.
            coord_noise_std: Standard deviation of coordinate noise augmentation.
            vocab_size: Vocabulary size for one-hot encoding.
            lazy_labels: If True, load label rows on demand instead of all at init.
            labels_chunk_size: Number of rows to read per chunk when scanning lazily.
        """

        def __init__(
            self,
            sequences_csv: str,
            labels_csv: str,
            max_crop_len: int = 500,
            coord_noise_std: float = 0.1,
            vocab_size: int = 5,
            lazy_labels: bool = False,
            labels_chunk_size: int = 10000,
        ) -> None:
            super().__init__()
            from .data_utils import load_sequences_csv, load_labels_csv

            self.max_crop_len = max_crop_len
            self.coord_noise_std = coord_noise_std
            self.vocab_size = vocab_size
            self._nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
            self._lazy_labels = lazy_labels
            self._labels_csv = labels_csv
            self._labels_chunk_size = labels_chunk_size

            # Always load sequences (small)
            seq_records = load_sequences_csv(sequences_csv)
            self._seq_index: Dict[str, str] = {rec.seq_id: rec.sequence for rec in seq_records}

            if not lazy_labels:
                # Original behaviour: load all labels into RAM
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
            else:
                # Lazy mode: only record which seq_ids exist in labels (scan CSV header)
                self.entries = []
                self._seq_ids_with_labels: List[str] = []
                labeled_ids = self._scan_labeled_ids()
                for rec in seq_records:
                    if rec.seq_id in labeled_ids:
                        self._seq_ids_with_labels.append(rec.seq_id)
                        # entries stores (seq_id, sequence, None) — coords loaded lazily
                        self.entries.append((rec.seq_id, rec.sequence, None))  # type: ignore[arg-type]
                logger.info(
                    "competition_dataset_lazy_init",
                    n_entries=len(self.entries),
                )

        def _scan_labeled_ids(self) -> set:
            """Quickly scan the labels CSV to find which seq_ids have labels."""
            labeled: set = set()
            try:
                with open(self._labels_csv, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        return labeled
                    # First column expected to be seq_id or ID-like field
                    for row in reader:
                        if row:
                            labeled.add(row[0])
            except Exception as e:
                logger.warning("labels_scan_error", error=str(e))
            return labeled

        def _load_labels_for_id(self, seq_id: str) -> Optional[np.ndarray]:
            """Load label coordinates for a single seq_id from the CSV.

            Uses chunked reading to avoid loading the full file into RAM.
            """
            try:
                import pandas as pd
                for chunk in pd.read_csv(
                    self._labels_csv,
                    chunksize=self._labels_chunk_size,
                ):
                    # Normalise column names (ensure strings)
                    chunk.columns = [str(c).strip() for c in chunk.columns]
                    id_col = chunk.columns[0]
                    match = chunk[chunk[id_col] == seq_id]
                    if not match.empty:
                        coord_cols = [c for c in chunk.columns if c != id_col]
                        coords = match[coord_cols].values.astype(np.float32)
                        # Expected shape: (L, 3) — already 2-D
                        if coords.ndim == 2 and coords.shape[1] == 3:
                            return coords
                        # Flat 2-D row or 1-D array: reshape to (L, 3)
                        flat = coords.ravel()
                        if flat.size % 3 == 0 and flat.size > 0:
                            return flat.reshape(-1, 3)
            except ImportError:
                # pandas not available — fall back to csv module
                try:
                    with open(self._labels_csv, newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header is None:
                            return None
                        for row in reader:
                            if row and row[0] == seq_id:
                                vals = [float(v) for v in row[1:] if v]
                                if len(vals) % 3 == 0 and vals:
                                    return np.array(vals, dtype=np.float32).reshape(-1, 3)
                except Exception as e2:
                    logger.debug("labels_csv_load_error", seq_id=seq_id, error=str(e2))
            except Exception as e:
                logger.debug("labels_chunk_load_error", seq_id=seq_id, error=str(e))
            return None

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, idx: int) -> Dict:
            """Get a single training example.

            Returns:
                Dict with keys: seq_onehot, coords, length, seq_id.
            """
            seq_id, sequence, coords = self.entries[idx]

            # In lazy mode, coords may be None — load on demand
            if coords is None:
                coords = self._load_labels_for_id(seq_id)
                if coords is None:
                    # Return zero coords as fallback
                    coords = np.zeros((len(sequence), 3), dtype=np.float32)

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


class DiskCacheManager:
    """Manages a bounded on-disk cache of downloaded dataset files.

    Tracks downloaded files and their sizes.  When the total cache size
    exceeds ``max_disk_cache_gb`` the oldest files are evicted.

    Supports two download backends:
    - ``"kaggle"``: uses the Kaggle CLI (``kaggle datasets download``).
    - ``"local"``: treats ``source_dir`` as a local directory and copies files.

    Args:
        cache_dir: Local directory used to store cached files.
        max_disk_cache_gb: Maximum total on-disk cache size in GB.
        dataset: Kaggle dataset identifier (``owner/name``).
        data_source: ``"kaggle"`` or ``"local"``.
        source_dir: Path to local data directory (used when data_source="local").
    """

    def __init__(
        self,
        cache_dir: str = "/tmp/rna_disk_cache",
        max_disk_cache_gb: float = 8.0,
        dataset: str = "stanford-rna-3d-folding",
        data_source: str = "kaggle",
        source_dir: str = "",
    ) -> None:
        self.cache_dir = cache_dir
        self.max_disk_cache_bytes = int(max_disk_cache_gb * (1024 ** 3))
        self.dataset = dataset
        self.data_source = data_source
        self.source_dir = source_dir

        os.makedirs(cache_dir, exist_ok=True)

        # Ordered list of (path, size_bytes, access_time) for eviction
        self._registry: List[Tuple[str, int, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_file(self, filename: str) -> Optional[str]:
        """Return the local path to *filename*, downloading it if necessary.

        Args:
            filename: Relative filename within the dataset (e.g. ``"PDB_RNA/1ABC.cif"``).

        Returns:
            Absolute local path, or None if the file could not be obtained.
        """
        local_path = os.path.join(self.cache_dir, os.path.basename(filename))

        if os.path.isfile(local_path):
            self._touch(local_path)
            return local_path

        # Download / copy
        success = self._fetch(filename, local_path)
        if not success or not os.path.isfile(local_path):
            return None

        size = os.path.getsize(local_path)
        self._registry.append((local_path, size, time.time()))
        self._evict_if_needed()
        return local_path

    def release_file(self, local_path: str) -> None:
        """Delete *local_path* and remove it from the registry.

        Call this after processing a file to keep disk usage low.
        """
        try:
            os.remove(local_path)
        except OSError:
            pass
        self._registry = [(p, s, t) for p, s, t in self._registry if p != local_path]

    def cleanup(self) -> None:
        """Delete all cached files and clear the registry."""
        for path, _size, _t in self._registry:
            try:
                os.remove(path)
            except OSError:
                pass
        self._registry = []
        # Also remove anything left in cache_dir
        try:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch(self, filename: str, dest: str) -> bool:
        """Download or copy *filename* to *dest*."""
        if self.data_source == "local":
            return self._copy_local(filename, dest)
        return self._download_kaggle(filename, dest)

    def _copy_local(self, filename: str, dest: str) -> bool:
        """Copy file from a local source directory."""
        src = os.path.join(self.source_dir, filename)
        if not os.path.isfile(src):
            # Try just the basename inside source_dir
            src = os.path.join(self.source_dir, os.path.basename(filename))
        if not os.path.isfile(src):
            logger.debug("local_file_not_found", filename=filename)
            return False
        try:
            shutil.copy2(src, dest)
            return True
        except Exception as e:
            logger.debug("local_copy_error", filename=filename, error=str(e))
            return False

    def _download_kaggle(self, filename: str, dest: str) -> bool:
        """Download a single file from the Kaggle dataset using the CLI."""
        try:
            result = subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", self.dataset,
                    "-f", filename,
                    "-p", self.cache_dir,
                    "--unzip",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.debug(
                    "kaggle_download_failed",
                    filename=filename,
                    stderr=result.stderr[:200],
                )
                return False
            # Kaggle puts file at cache_dir/<basename>
            basename = os.path.basename(filename)
            downloaded = os.path.join(self.cache_dir, basename)
            if os.path.isfile(downloaded) and downloaded != dest:
                shutil.move(downloaded, dest)
            return os.path.isfile(dest)
        except Exception as e:
            logger.debug("kaggle_download_error", filename=filename, error=str(e))
            return False

    def _touch(self, local_path: str) -> None:
        """Update the access timestamp for *local_path* in the registry."""
        now = time.time()
        self._registry = [
            (p, s, now if p == local_path else t)
            for p, s, t in self._registry
        ]

    def _evict_if_needed(self) -> None:
        """Evict oldest files until total cache size is within budget."""
        total = sum(s for _, s, _ in self._registry)
        if total <= self.max_disk_cache_bytes:
            return

        # Sort by access time ascending (oldest first)
        self._registry.sort(key=lambda x: x[2])
        while self._registry and total > self.max_disk_cache_bytes:
            path, size, _ = self._registry.pop(0)
            try:
                os.remove(path)
            except OSError:
                pass
            total -= size
            logger.debug("disk_cache_evicted", path=path, freed_mb=f"{size / 1e6:.1f}")


if _TORCH_AVAILABLE:

    class StreamingCIFDataset(IterableDataset):
        """Memory-safe CIF dataset that downloads files on demand and deletes after use.

        Instead of requiring all CIF files on disk, it accepts a list of
        filenames and uses a ``DiskCacheManager`` to fetch each chunk from the
        configured data source (Kaggle or local), then deletes the files once
        they have been parsed.

        Args:
            cif_filenames: List of CIF filenames (basenames or relative paths).
            disk_cache: ``DiskCacheManager`` instance.
            chunk_size: Number of CIF files to download and process per chunk.
            parse_fn: Function mapping a file path to an RNAStructure (or None).
            max_crop_len: Maximum sequence length; longer sequences are cropped.
            bucket_multiple: Pad lengths to multiples of this value.
        """

        def __init__(
            self,
            cif_filenames: List[str],
            disk_cache: "DiskCacheManager",
            chunk_size: int = 20,
            parse_fn: Optional[Callable] = None,
            max_crop_len: int = 500,
            bucket_multiple: int = 32,
        ) -> None:
            super().__init__()
            self.cif_filenames = list(cif_filenames)
            self.disk_cache = disk_cache
            self.chunk_size = chunk_size
            self.max_crop_len = max_crop_len
            self.bucket_multiple = bucket_multiple

            if parse_fn is None:
                from .data_utils import parse_cif_c3_coords
                self.parse_fn = parse_cif_c3_coords
            else:
                self.parse_fn = parse_fn

        def __iter__(self) -> Iterator:
            """Download chunks, parse, yield, then delete downloaded files."""
            filenames = self.cif_filenames.copy()
            random.shuffle(filenames)

            for chunk_start in range(0, len(filenames), self.chunk_size):
                chunk_files = filenames[chunk_start:chunk_start + self.chunk_size]
                structures = []
                local_paths: List[str] = []

                for fname in chunk_files:
                    local_path = self.disk_cache.ensure_file(fname)
                    if local_path is None:
                        logger.debug("streaming_cif_skip_no_file", filename=fname)
                        continue
                    local_paths.append(local_path)
                    try:
                        structure = self.parse_fn(local_path)
                        if structure is not None:
                            structures.append(structure)
                    except Exception as e:
                        logger.debug("streaming_cif_parse_skip", file=fname, error=str(e))

                # Delete downloaded files immediately after parsing
                for lp in local_paths:
                    self.disk_cache.release_file(lp)

                # Length bucketing within chunk
                structures.sort(key=lambda s: len(s.sequence))

                for structure in structures:
                    seq = structure.sequence
                    coords = structure.coords_c3
                    L = len(seq)

                    if L > self.max_crop_len:
                        start = random.randint(0, L - self.max_crop_len)
                        seq = seq[start:start + self.max_crop_len]
                        coords = coords[start:start + self.max_crop_len]
                        L = self.max_crop_len

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

                del structures
                gc.collect()

    class StreamingMSALoader:
        """MSA loader that downloads files on demand and deletes after parsing.

        Wraps ``DiskCacheManager`` so that MSA files are fetched from Kaggle (or
        a local source) only when needed, then immediately removed to keep disk
        usage low.

        Args:
            disk_cache: ``DiskCacheManager`` instance.
            msa_subdir: Sub-path within the Kaggle dataset for MSA files (e.g. ``"MSA"``).
            max_seqs: Maximum number of sequences to keep after subsampling.
            vocab_size: Vocabulary size for one-hot encoding.
        """

        def __init__(
            self,
            disk_cache: "DiskCacheManager",
            msa_subdir: str = "MSA",
            max_seqs: int = 128,
            vocab_size: int = 5,
        ) -> None:
            self.disk_cache = disk_cache
            self.msa_subdir = msa_subdir
            self.max_seqs = max_seqs
            self.vocab_size = vocab_size
            self._nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "-": 4}

        def load(self, seq_id: str) -> Optional["torch.Tensor"]:
            """Download, parse, and delete MSA file for *seq_id*.

            Args:
                seq_id: Sequence identifier.

            Returns:
                One-hot MSA tensor ``(N, L, V)`` or None if unavailable.
            """
            for ext in [".a3m", ".sto", ".fasta"]:
                remote_name = os.path.join(self.msa_subdir, seq_id + ext)
                local_path = self.disk_cache.ensure_file(remote_name)
                if local_path is not None:
                    result = self._parse_msa(local_path)
                    self.disk_cache.release_file(local_path)
                    return result
            return None

        def _parse_msa(self, path: str) -> Optional["torch.Tensor"]:
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

                if len(sequences) > self.max_seqs:
                    indices = random.sample(range(len(sequences)), self.max_seqs)
                    sequences = [sequences[i] for i in sorted(indices)]

                max_len = max(len(s) for s in sequences)
                onehot = torch.zeros(len(sequences), max_len, self.vocab_size)
                for i, seq in enumerate(sequences):
                    for j, c in enumerate(seq):
                        idx = self._nuc_map.get(c, 4)
                        onehot[i, j, idx] = 1.0

                return onehot

            except Exception as e:
                logger.debug("streaming_msa_parse_error", path=path, error=str(e))
                return None
            finally:
                gc.collect()



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
