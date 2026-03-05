"""Complete training script for Colab T4. Three phases: pretrain → finetune → MSA.

Phase 1: Pre-train on CIF structures from PDB_RNA (ChunkedCIFDataset)
Phase 2: Fine-tune on competition CSV data (CompetitionDataset)
Phase 3: Add MSA features (MSAChunkLoader)

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

        def phase1_pretrain(self, cif_dir: str, n_epochs: Optional[int] = None) -> None:
            """Phase 1: Pre-train on CIF structures.

            Args:
                cif_dir: Directory containing PDB RNA CIF files.
                n_epochs: Override number of epochs.
            """
            n_epochs = n_epochs or self.train_cfg.phase1_epochs
            logger.info("phase1_start", cif_dir=cif_dir, n_epochs=n_epochs)

            patch_dataloader_memory()
            dataset = ChunkedCIFDataset(
                cif_dir, chunk_size=self.mem_cfg.cif_chunk_size,
                max_crop_len=self.train_cfg.max_crop_len,
            )
            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            for epoch in range(n_epochs):
                self.train_epoch(dataloader, epoch)
                self.vram_monitor.log_status()

            logger.info("phase1_done")

        def phase2_finetune(
            self,
            train_csv: str,
            labels_csv: str,
            val_csv: Optional[str] = None,
            val_labels: Optional[str] = None,
            n_epochs: Optional[int] = None,
        ) -> None:
            """Phase 2: Fine-tune on competition CSV data.

            Args:
                train_csv: Path to train_sequences.csv.
                labels_csv: Path to train_labels.csv.
                val_csv: Optional validation sequences CSV.
                val_labels: Optional validation labels CSV.
                n_epochs: Override number of epochs.
            """
            n_epochs = n_epochs or self.train_cfg.phase2_epochs
            logger.info("phase2_start", n_epochs=n_epochs)

            dataset = CompetitionDataset(
                train_csv, labels_csv,
                max_crop_len=self.train_cfg.max_crop_len,
                coord_noise_std=self.train_cfg.coord_noise_std,
            )

            lengths = [len(e[1]) for e in dataset.entries]
            sampler = BucketBatchSampler(
                lengths, bucket_multiple=self.mem_cfg.length_bucket_multiple,
            )
            dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

            for epoch in range(n_epochs):
                self.train_epoch(dataloader, epoch)

                # Validate periodically
                if val_csv and val_labels and (epoch + 1) % 10 == 0:
                    val_dataset = CompetitionDataset(val_csv, val_labels)
                    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
                    tm = self.validate(val_loader)
                    if tm > self.best_tm_score:
                        self.best_tm_score = tm
                        self._save_checkpoint("best_model.pt")
                    del val_dataset, val_loader
                    gc.collect()

            logger.info("phase2_done", best_tm=f"{self.best_tm_score:.4f}")

        def phase3_msa(
            self,
            train_csv: str,
            labels_csv: str,
            msa_dir: str,
            n_epochs: Optional[int] = None,
        ) -> None:
            """Phase 3: Train with MSA features.

            Args:
                train_csv: Path to train_sequences.csv.
                labels_csv: Path to train_labels.csv.
                msa_dir: Directory containing MSA files.
                n_epochs: Override number of epochs.
            """
            n_epochs = n_epochs or self.train_cfg.phase3_epochs
            logger.info("phase3_start", msa_dir=msa_dir, n_epochs=n_epochs)

            dataset = CompetitionDataset(
                train_csv, labels_csv,
                max_crop_len=self.train_cfg.max_crop_len,
            )
            msa_loader = MSAChunkLoader(msa_dir, max_seqs=self.train_cfg.msa_max_seqs)

            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            self.model.train()
            for epoch in range(n_epochs):
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

                avg_loss = total_loss / max(n_batches, 1)
                logger.info("phase3_epoch", epoch=epoch, avg_loss=f"{avg_loss:.4f}")

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

        def _save_checkpoint(self, filename: str) -> None:
            """Save model checkpoint.

            Args:
                filename: Checkpoint filename.
            """
            path = os.path.join(self.config.output_dir, filename)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_tm_score": self.best_tm_score,
            }, path)
            logger.info("checkpoint_saved", path=path)


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
