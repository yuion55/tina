"""Complete training script for Colab T4. Three phases: pretrain → finetune → MSA.

Phase 1: Pre-train on CIF structures from PDB_RNA (ChunkedCIFDataset or StreamingCIFDataset)
Phase 2: Fine-tune on competition CSV data (CompetitionDataset)
Phase 3: Add MSA features (MSAChunkLoader or StreamingMSALoader)

Uses FP16 via ``torch.cuda.amp.autocast``, gradient accumulation (8 steps),
and aggressive memory management throughout.
"""

from __future__ import annotations

import gc
import os
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch_not_available", msg="PyTorch not installed; training unavailable")


def _get_ram_used_gb() -> float:
    """Return current process RAM usage in GB.

    Uses ``psutil`` if available, falls back to ``/proc/meminfo``.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        pass
    try:
        total_kb = 0
        avail_kb = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
        if total_kb > 0:
            return (total_kb - avail_kb) / (1024 ** 2)
    except Exception:
        pass
    return 0.0



if _TORCH_AVAILABLE:
    from .config import PipelineConfig, PhysicsNetConfig, TrainingConfig, MemoryConfig
    from .rna_physicsnet import RNAPhysicsNet, get_dynamic_config
    from .physics_loss import PhysicsInformedLoss, coords_to_distogram_target
    from .memory_manager import (
        ChunkedCIFDataset,
        CompetitionDataset,
        MSAChunkLoader,
        VRAMMonitor,
        BucketBatchSampler,
        DiskCacheManager,
        StreamingCIFDataset,
        StreamingMSALoader,
        patch_dataloader_memory,
    )
    from .scoring import compute_tm_score

    class Trainer:
        """3-phase trainer for RNA-PhysicsNet on Colab T4.

        Args:
            config: Pipeline configuration.
            device: PyTorch device (auto-detected if None).
        """

        def __init__(self, config: Optional[PipelineConfig] = None,
                     device: Optional[str] = None) -> None:
            self.config = config or PipelineConfig()
            self.train_cfg = self.config.training
            self.mem_cfg = self.config.memory

            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.vram_monitor = VRAMMonitor()
            self.best_tm_score = 0.0
            self._start_epoch: int = 0  # for checkpoint resume

            # Build model (default config for medium sequences)
            dyn_cfg = get_dynamic_config(300, self.mem_cfg.vram_gb)
            self.model = RNAPhysicsNet(dyn_cfg, self.config.physics_net).to(self.device)
            self.loss_fn = PhysicsInformedLoss(self.config.biology)

            # Optimizer + scheduler + scaler
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.train_cfg.lr_peak,
                weight_decay=0.01,
            )
            self.scaler = GradScaler()

            logger.info(
                "trainer_init",
                device=str(self.device),
                n_params=sum(p.numel() for p in self.model.parameters()),
            )

        # ------------------------------------------------------------------
        # RAM monitoring
        # ------------------------------------------------------------------

        def _check_ram(self) -> None:
            """Emergency cleanup if RAM usage exceeds ``mem_cfg.ram_limit_gb``."""
            used_gb = _get_ram_used_gb()
            if used_gb > self.mem_cfg.ram_limit_gb:
                logger.warning(
                    "ram_limit_exceeded",
                    used_gb=f"{used_gb:.2f}",
                    limit_gb=self.mem_cfg.ram_limit_gb,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Checkpoint helpers
        # ------------------------------------------------------------------

        def load_checkpoint(self, path: str) -> None:
            """Resume training from a saved checkpoint.

            Args:
                path: Path to a ``.pt`` checkpoint file saved by ``_save_checkpoint``.
            """
            if not os.path.isfile(path):
                logger.warning("checkpoint_not_found", path=path)
                return
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_tm_score = checkpoint.get("best_tm_score", 0.0)
            self._start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info(
                "checkpoint_loaded",
                path=path,
                resume_epoch=self._start_epoch,
                best_tm=f"{self.best_tm_score:.4f}",
            )


        def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> float:
            """Train for one epoch with gradient accumulation.

            Args:
                dataloader: Training dataloader.
                epoch: Current epoch number.

            Returns:
                Average loss for the epoch.
            """
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                seq_onehot = batch["seq_onehot"].squeeze(0).to(self.device)
                true_coords = batch["coords"].squeeze(0).to(self.device)

                with autocast():
                    pred_coords, pred_disto, plddt = self.model(seq_onehot)

                    # Compute targets
                    true_disto = coords_to_distogram_target(true_coords)
                    sequence = seq_onehot.argmax(dim=-1)

                    loss = self.loss_fn(
                        pred_coords, true_coords,
                        pred_disto, true_disto,
                        sequence=sequence,
                    )
                    loss = loss / self.train_cfg.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.train_cfg.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.train_cfg.grad_accum_steps
                n_batches += 1

                # Memory cleanup
                if batch_idx % self.mem_cfg.gc_every_n_batches == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._check_ram()

            avg_loss = total_loss / max(n_batches, 1)
            logger.info("epoch_done", epoch=epoch, avg_loss=f"{avg_loss:.4f}")
            return avg_loss

        def validate(self, dataloader: DataLoader) -> float:
            """Compute validation TM-score.

            Args:
                dataloader: Validation dataloader.

            Returns:
                Mean TM-score across validation set.
            """
            self.model.eval()
            tm_scores = []

            with torch.no_grad():
                for batch in dataloader:
                    seq_onehot = batch["seq_onehot"].squeeze(0).to(self.device)
                    true_coords = batch["coords"].squeeze(0)

                    with autocast():
                        pred_coords, _, _ = self.model(seq_onehot)

                    pred_np = pred_coords.cpu().numpy()
                    true_np = true_coords.numpy()
                    L = min(pred_np.shape[0], true_np.shape[0])

                    if L > 0:
                        tm = compute_tm_score(pred_np[:L], true_np[:L])
                        tm_scores.append(tm)

            mean_tm = float(np.mean(tm_scores)) if tm_scores else 0.0
            logger.info("validation_done", mean_tm_score=f"{mean_tm:.4f}", n_seqs=len(tm_scores))
            return mean_tm

        def phase1_pretrain(
            self,
            cif_dir: str,
            n_epochs: Optional[int] = None,
            cif_filenames: Optional[list] = None,
            resume_checkpoint: Optional[str] = None,
        ) -> None:
            """Phase 1: Pre-train on CIF structures.

            When ``mem_cfg.enable_streaming`` is True and ``cif_filenames`` is
            provided (a list of CIF filenames to stream from the configured data
            source), a :class:`StreamingCIFDataset` is used instead of the
            local :class:`ChunkedCIFDataset`.

            Args:
                cif_dir: Directory containing PDB RNA CIF files (used when
                    streaming is disabled or ``cif_filenames`` is None).
                n_epochs: Override number of epochs.
                cif_filenames: Optional list of CIF filenames for streaming mode.
                    Each entry should be a relative path within the Kaggle dataset
                    (e.g. ``"PDB_RNA/1ABC.cif"``).
                resume_checkpoint: Optional path to a checkpoint to resume from.
            """
            n_epochs = n_epochs or self.train_cfg.phase1_epochs
            logger.info("phase1_start", cif_dir=cif_dir, n_epochs=n_epochs)

            if resume_checkpoint:
                self.load_checkpoint(resume_checkpoint)

            patch_dataloader_memory()

            if self.mem_cfg.enable_streaming and cif_filenames:
                disk_cache = DiskCacheManager(
                    max_disk_cache_gb=self.mem_cfg.max_disk_cache_gb,
                    dataset=self.mem_cfg.kaggle_dataset,
                    data_source=self.mem_cfg.data_source,
                    source_dir=cif_dir,
                )
                dataset = StreamingCIFDataset(
                    cif_filenames,
                    disk_cache,
                    chunk_size=self.mem_cfg.streaming_chunk_size,
                    max_crop_len=self.train_cfg.max_crop_len,
                )
                logger.info(
                    "phase1_streaming",
                    n_files=len(cif_filenames),
                    chunk_size=self.mem_cfg.streaming_chunk_size,
                )
            else:
                dataset = ChunkedCIFDataset(
                    cif_dir, chunk_size=self.mem_cfg.cif_chunk_size,
                    max_crop_len=self.train_cfg.max_crop_len,
                )

            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            start = self._start_epoch
            for epoch in range(start, n_epochs):
                self.train_epoch(dataloader, epoch)
                self.vram_monitor.log_status()
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(f"phase1_epoch{epoch + 1}.pt", epoch=epoch)

            logger.info("phase1_done")

        def phase2_finetune(
            self,
            train_csv: str,
            labels_csv: str,
            val_csv: Optional[str] = None,
            val_labels: Optional[str] = None,
            n_epochs: Optional[int] = None,
            resume_checkpoint: Optional[str] = None,
        ) -> None:
            """Phase 2: Fine-tune on competition CSV data.

            Args:
                train_csv: Path to train_sequences.csv.
                labels_csv: Path to train_labels.csv.
                val_csv: Optional validation sequences CSV.
                val_labels: Optional validation labels CSV.
                n_epochs: Override number of epochs.
                resume_checkpoint: Optional path to a checkpoint to resume from.
            """
            n_epochs = n_epochs or self.train_cfg.phase2_epochs
            logger.info("phase2_start", n_epochs=n_epochs)

            if resume_checkpoint:
                self.load_checkpoint(resume_checkpoint)

            lazy = self.mem_cfg.enable_streaming
            dataset = CompetitionDataset(
                train_csv, labels_csv,
                max_crop_len=self.train_cfg.max_crop_len,
                coord_noise_std=self.train_cfg.coord_noise_std,
                lazy_labels=lazy,
            )

            lengths = [len(e[1]) for e in dataset.entries]
            sampler = BucketBatchSampler(
                lengths, bucket_multiple=self.mem_cfg.length_bucket_multiple,
            )
            dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

            start = self._start_epoch
            for epoch in range(start, n_epochs):
                self.train_epoch(dataloader, epoch)

                # Validate periodically
                if val_csv and val_labels and (epoch + 1) % 10 == 0:
                    val_dataset = CompetitionDataset(val_csv, val_labels)
                    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
                    tm = self.validate(val_loader)
                    if tm > self.best_tm_score:
                        self.best_tm_score = tm
                        self._save_checkpoint("best_model.pt", epoch=epoch)
                    del val_dataset, val_loader
                    gc.collect()

            logger.info("phase2_done", best_tm=f"{self.best_tm_score:.4f}")

        def phase3_msa(
            self,
            train_csv: str,
            labels_csv: str,
            msa_dir: str,
            n_epochs: Optional[int] = None,
            resume_checkpoint: Optional[str] = None,
        ) -> None:
            """Phase 3: Train with MSA features.

            When ``mem_cfg.enable_streaming`` is True a :class:`StreamingMSALoader`
            is used to fetch MSA files on demand and delete them after parsing.

            Args:
                train_csv: Path to train_sequences.csv.
                labels_csv: Path to train_labels.csv.
                msa_dir: Directory containing MSA files (used in local mode).
                n_epochs: Override number of epochs.
                resume_checkpoint: Optional path to a checkpoint to resume from.
            """
            n_epochs = n_epochs or self.train_cfg.phase3_epochs
            logger.info("phase3_start", msa_dir=msa_dir, n_epochs=n_epochs)

            if resume_checkpoint:
                self.load_checkpoint(resume_checkpoint)

            dataset = CompetitionDataset(
                train_csv, labels_csv,
                max_crop_len=self.train_cfg.max_crop_len,
            )

            if self.mem_cfg.enable_streaming:
                disk_cache = DiskCacheManager(
                    max_disk_cache_gb=self.mem_cfg.max_disk_cache_gb,
                    dataset=self.mem_cfg.kaggle_dataset,
                    data_source=self.mem_cfg.data_source,
                    source_dir=msa_dir,
                )
                msa_loader: "MSAChunkLoader | StreamingMSALoader" = StreamingMSALoader(
                    disk_cache,
                    msa_subdir="MSA",
                    max_seqs=self.train_cfg.msa_max_seqs,
                )
                logger.info("phase3_streaming_msa")
            else:
                msa_loader = MSAChunkLoader(msa_dir, max_seqs=self.train_cfg.msa_max_seqs)

            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            self.model.train()
            start = self._start_epoch
            for epoch in range(start, n_epochs):
                total_loss = 0.0
                n_batches = 0
                self.optimizer.zero_grad()

                for batch_idx, batch in enumerate(dataloader):
                    seq_onehot = batch["seq_onehot"].squeeze(0).to(self.device)
                    true_coords = batch["coords"].squeeze(0).to(self.device)

                    # Load MSA if available
                    msa_feat = msa_loader.load(batch["seq_id"][0])
                    if msa_feat is not None:
                        msa_feat = msa_feat.to(self.device)

                    with autocast():
                        pred_coords, pred_disto, plddt = self.model(
                            seq_onehot, msa_feat=msa_feat
                        )
                        true_disto = coords_to_distogram_target(true_coords)
                        sequence = seq_onehot.argmax(dim=-1)
                        loss = self.loss_fn(
                            pred_coords, true_coords, pred_disto, true_disto,
                            sequence=sequence,
                        )
                        loss = loss / self.train_cfg.grad_accum_steps

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.train_cfg.grad_accum_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    total_loss += loss.item() * self.train_cfg.grad_accum_steps
                    n_batches += 1

                    # Cleanup
                    del msa_feat
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._check_ram()

                avg_loss = total_loss / max(n_batches, 1)
                logger.info("phase3_epoch", epoch=epoch, avg_loss=f"{avg_loss:.4f}")
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(f"phase3_epoch{epoch + 1}.pt", epoch=epoch)

            logger.info("phase3_done")

        def export_onnx(self, path: Optional[str] = None) -> str:
            """Export model to ONNX format with dynamic axes.

            Args:
                path: Output ONNX file path.

            Returns:
                Path to exported ONNX file.
            """
            if path is None:
                path = self.config.onnx_model_path

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            self.model.eval()
            self.model.cpu()

            dummy_len = 50
            vocab_size = self.config.physics_net.vocab_size
            dummy_input = torch.zeros(dummy_len, vocab_size)
            dummy_input[:, 0] = 1.0  # All A's

            torch.onnx.export(
                self.model,
                (dummy_input,),
                path,
                input_names=["seq_onehot"],
                output_names=["coords", "distogram", "plddt"],
                dynamic_axes={
                    "seq_onehot": {0: "seq_len"},
                    "coords": {0: "seq_len"},
                    "distogram": {0: "seq_len", 1: "seq_len"},
                    "plddt": {0: "seq_len"},
                },
                opset_version=17,
            )

            self.model.to(self.device)
            logger.info("onnx_exported", path=path)
            return path

        def _save_checkpoint(self, filename: str, epoch: int = 0) -> None:
            """Save model checkpoint.

            Args:
                filename: Checkpoint filename.
                epoch: Current epoch number (stored for resume support).
            """
            path = os.path.join(self.config.output_dir, filename)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_tm_score": self.best_tm_score,
                "epoch": epoch,
            }, path)
            logger.info("checkpoint_saved", path=path, epoch=epoch)


def main() -> None:
    """Run full 3-phase training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="RNA-PhysicsNet Training")
    parser.add_argument("--cif-dir", default="/kaggle/input/stanford-rna-3d-folding/PDB_RNA")
    parser.add_argument("--train-csv", default="/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
    parser.add_argument("--labels-csv", default="/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
    parser.add_argument("--val-csv", default="/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
    parser.add_argument("--val-labels", default="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
    parser.add_argument("--msa-dir", default="/kaggle/input/stanford-rna-3d-folding/MSA")
    parser.add_argument("--onnx-out", default="topomatrix_rna/models/rna_physicsnet.onnx")
    args = parser.parse_args()

    config = PipelineConfig()
    trainer = Trainer(config)

    # Phase 1
    if os.path.isdir(args.cif_dir):
        trainer.phase1_pretrain(args.cif_dir)

    # Phase 2
    if os.path.isfile(args.train_csv):
        trainer.phase2_finetune(
            args.train_csv, args.labels_csv,
            args.val_csv, args.val_labels,
        )

    # Phase 3
    if os.path.isdir(args.msa_dir):
        trainer.phase3_msa(args.train_csv, args.labels_csv, args.msa_dir)

    # Export
    trainer.export_onnx(args.onnx_out)
    logger.info("training_complete")


if __name__ == "__main__":
    main()
