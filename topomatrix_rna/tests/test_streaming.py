"""Tests for streaming / on-demand data classes in memory_manager and config."""

import csv
import os
import shutil
import tempfile

import numpy as np
import pytest

from topomatrix_rna.config import MemoryConfig


class TestMemoryConfigStreaming:
    """Verify new streaming fields are present in MemoryConfig."""

    def test_default_values(self):
        cfg = MemoryConfig()
        assert cfg.enable_streaming is True
        assert cfg.max_disk_cache_gb == 8.0
        assert cfg.streaming_chunk_size == 20
        assert cfg.kaggle_dataset == "stanford-rna-3d-folding"
        assert cfg.data_source == "kaggle"
        assert cfg.ram_limit_gb == 10.0

    def test_custom_values(self):
        cfg = MemoryConfig(
            enable_streaming=False,
            max_disk_cache_gb=4.0,
            streaming_chunk_size=5,
            data_source="local",
        )
        assert cfg.enable_streaming is False
        assert cfg.max_disk_cache_gb == 4.0
        assert cfg.streaming_chunk_size == 5
        assert cfg.data_source == "local"


class TestDiskCacheManager:
    """Tests for DiskCacheManager."""

    def test_import(self):
        from topomatrix_rna.memory_manager import DiskCacheManager
        assert DiskCacheManager is not None

    def test_init_creates_cache_dir(self):
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(cache_dir=cache_dir)
            assert os.path.isdir(cache_dir)

    def test_local_copy_existing_file(self):
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a source file
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            src_file = os.path.join(src_dir, "test.cif")
            with open(src_file, "w") as f:
                f.write("data_TEST\n")

            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(
                cache_dir=cache_dir,
                data_source="local",
                source_dir=src_dir,
            )
            path = mgr.ensure_file("test.cif")
            assert path is not None
            assert os.path.isfile(path)
            with open(path) as f:
                assert "data_TEST" in f.read()

    def test_local_copy_missing_file(self):
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(
                cache_dir=cache_dir,
                data_source="local",
                source_dir=tmpdir,
            )
            path = mgr.ensure_file("nonexistent.cif")
            assert path is None

    def test_release_file_deletes(self):
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            src_file = os.path.join(src_dir, "test.cif")
            with open(src_file, "w") as f:
                f.write("data_TEST\n")

            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(cache_dir=cache_dir, data_source="local", source_dir=src_dir)
            path = mgr.ensure_file("test.cif")
            assert path is not None and os.path.isfile(path)

            mgr.release_file(path)
            assert not os.path.isfile(path)
            # Registry should be empty
            assert len(mgr._registry) == 0

    def test_cleanup_removes_all(self):
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            for i in range(3):
                with open(os.path.join(src_dir, f"file{i}.cif"), "w") as f:
                    f.write(f"data_{i}\n")

            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(cache_dir=cache_dir, data_source="local", source_dir=src_dir)
            paths = [mgr.ensure_file(f"file{i}.cif") for i in range(3)]
            assert all(p is not None and os.path.isfile(p) for p in paths)

            mgr.cleanup()
            assert not os.path.isdir(cache_dir) or not any(
                os.path.isfile(p) for p in paths if p
            )

    def test_eviction_when_cache_full(self):
        """Files should be evicted when cache exceeds max_disk_cache_gb."""
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            # Create files of known size
            for i in range(5):
                with open(os.path.join(src_dir, f"big{i}.dat"), "wb") as f:
                    f.write(b"X" * 1024)  # 1 KB each

            cache_dir = os.path.join(tmpdir, "cache")
            # Allow only 3 KB (= 3 / 1024**2 GB)
            mgr = DiskCacheManager(
                cache_dir=cache_dir,
                data_source="local",
                source_dir=src_dir,
                max_disk_cache_gb=3 * 1024 / (1024 ** 3),  # 3 KB in GB
            )
            for i in range(5):
                mgr.ensure_file(f"big{i}.dat")

            # Total in registry should be ≤ 3 KB
            total = sum(s for _, s, _ in mgr._registry)
            assert total <= 3 * 1024

    def test_touch_updates_timestamp(self):
        """Cache hit should update file's access time."""
        import time
        from topomatrix_rna.memory_manager import DiskCacheManager
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, "a.cif"), "w") as f:
                f.write("data\n")

            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(cache_dir=cache_dir, data_source="local", source_dir=src_dir)
            mgr.ensure_file("a.cif")
            t1 = mgr._registry[0][2]
            time.sleep(0.01)
            mgr.ensure_file("a.cif")  # cache hit → touch
            t2 = mgr._registry[0][2]
            assert t2 >= t1


class TestCompetitionDatasetLazy:
    """Tests for lazy label loading in CompetitionDataset."""

    def _make_csvs(self, tmpdir, n=5):
        """Create minimal sequences + labels CSVs."""
        seq_path = os.path.join(tmpdir, "seqs.csv")
        lbl_path = os.path.join(tmpdir, "labels.csv")

        seqs = [f"seq{i}" for i in range(n)]
        sequences = ["ACGUACGU" * (i + 1) for i in range(n)]

        with open(seq_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence"])
            for s, seq in zip(seqs, sequences):
                w.writerow([s, seq])

        # Labels: id, x1, y1, z1, x2, y2, z2, ...
        with open(lbl_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["id"] + [f"{a}{i}" for i in range(1, 9) for a in ["x", "y", "z"]]
            w.writerow(header)
            for s, seq in zip(seqs, sequences):
                row = [s] + [float(j) for j in range(len(seq) * 3)]
                w.writerow(row)

        return seq_path, lbl_path

    def test_lazy_false_loads_all(self):
        """Non-lazy mode should load all entries at init."""
        try:
            from topomatrix_rna.memory_manager import CompetitionDataset
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path, lbl_path = self._make_csvs(tmpdir)
            ds = CompetitionDataset(seq_path, lbl_path, lazy_labels=False)
            assert len(ds) == 5
            # coords should not be None
            for _, _, coords in ds.entries:
                assert coords is not None

    def test_lazy_true_defers_loading(self):
        """Lazy mode should init with None coords in entries."""
        try:
            from topomatrix_rna.memory_manager import CompetitionDataset
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path, lbl_path = self._make_csvs(tmpdir)
            ds = CompetitionDataset(seq_path, lbl_path, lazy_labels=True)
            assert len(ds) == 5
            # coords should all be None at init time
            for _, _, coords in ds.entries:
                assert coords is None

    def test_scan_labeled_ids(self):
        """_scan_labeled_ids should return the correct set of IDs."""
        try:
            from topomatrix_rna.memory_manager import CompetitionDataset
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path, lbl_path = self._make_csvs(tmpdir, n=3)
            ds = CompetitionDataset(seq_path, lbl_path, lazy_labels=True)
            ids = ds._scan_labeled_ids()
            assert ids == {"seq0", "seq1", "seq2"}

    def test_getitem_lazy_returns_dict(self):
        """__getitem__ on lazy dataset should load coords and return valid dict."""
        try:
            import torch
            from topomatrix_rna.memory_manager import CompetitionDataset
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path, lbl_path = self._make_csvs(tmpdir, n=2)
            ds = CompetitionDataset(seq_path, lbl_path, lazy_labels=True)
            item = ds[0]
            assert "seq_onehot" in item
            assert "coords" in item
            assert "length" in item
            assert "seq_id" in item
            assert item["seq_onehot"].shape[1] == 5  # vocab_size

    def test_getitem_non_lazy_returns_dict(self):
        """__getitem__ on non-lazy dataset should also work correctly."""
        try:
            import torch
            from topomatrix_rna.memory_manager import CompetitionDataset
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path, lbl_path = self._make_csvs(tmpdir, n=2)
            ds = CompetitionDataset(seq_path, lbl_path, lazy_labels=False)
            item = ds[0]
            assert "seq_onehot" in item
            assert item["coords"].dtype == torch.float32


class TestStreamingCIFDataset:
    """Tests for StreamingCIFDataset."""

    def test_import(self):
        try:
            from topomatrix_rna.memory_manager import StreamingCIFDataset
        except ImportError:
            pytest.skip("torch not available")
        assert StreamingCIFDataset is not None

    def test_empty_filenames_yields_nothing(self):
        try:
            from topomatrix_rna.memory_manager import StreamingCIFDataset, DiskCacheManager
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(cache_dir=cache_dir, data_source="local", source_dir=tmpdir)
            ds = StreamingCIFDataset([], mgr)
            items = list(ds)
            assert items == []

    def test_missing_files_skipped(self):
        """Files that cannot be fetched should be silently skipped."""
        try:
            from topomatrix_rna.memory_manager import StreamingCIFDataset, DiskCacheManager
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(cache_dir=cache_dir, data_source="local", source_dir=tmpdir)
            ds = StreamingCIFDataset(["ghost1.cif", "ghost2.cif"], mgr)
            items = list(ds)
            assert items == []


class TestStreamingMSALoader:
    """Tests for StreamingMSALoader."""

    def test_import(self):
        try:
            from topomatrix_rna.memory_manager import StreamingMSALoader
        except ImportError:
            pytest.skip("torch not available")
        assert StreamingMSALoader is not None

    def test_missing_msa_returns_none(self):
        try:
            from topomatrix_rna.memory_manager import StreamingMSALoader, DiskCacheManager
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            src_dir = os.path.join(tmpdir, "msa")
            os.makedirs(src_dir)
            mgr = DiskCacheManager(cache_dir=cache_dir, data_source="local", source_dir=src_dir)
            loader = StreamingMSALoader(mgr)
            result = loader.load("nonexistent_seq")
            assert result is None

    def test_load_fasta_file(self):
        """Should load a FASTA file, parse it, return one-hot tensor, delete file."""
        try:
            import torch
            from topomatrix_rna.memory_manager import StreamingMSALoader, DiskCacheManager
        except ImportError:
            pytest.skip("torch not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "MSA")
            os.makedirs(src_dir)
            fasta_content = ">seq1\nACGUACGU\n>seq2\nGCUAGCUA\n"
            with open(os.path.join(src_dir, "myseq.fasta"), "w") as f:
                f.write(fasta_content)

            cache_dir = os.path.join(tmpdir, "cache")
            mgr = DiskCacheManager(
                cache_dir=cache_dir,
                data_source="local",
                source_dir=tmpdir,  # source_dir is parent of MSA/
            )
            loader = StreamingMSALoader(mgr, msa_subdir="MSA")
            result = loader.load("myseq")
            assert result is not None
            assert result.shape == (2, 8, 5)
            # File should have been deleted from cache
            cached = os.path.join(cache_dir, "myseq.fasta")
            assert not os.path.isfile(cached)
