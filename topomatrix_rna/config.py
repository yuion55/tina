"""Configuration dataclasses for the TOPOMATRIX-RNA pipeline.

All hyperparameters are centralized here as frozen-capable dataclasses.
No hardcoded constants elsewhere in the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AtlasConfig:
    """Configuration for Stage 0: Topological Atlas Construction."""

    cif_dir: str = "/kaggle/input/stanford-rna-3d-folding/PDB_RNA"
    metadata_csv: str = "/kaggle/input/stanford-rna-3d-folding/extra/rna_metadata.csv"
    max_resolution_angstrom: float = 3.0
    n_landmarks: int = 500
    rips_max_radius: float = 10.0
    rips_step: float = 0.1
    persistence_image_size: int = 50
    persistence_sigma: float = 0.2
    stable_rank_dims: int = 64
    atlas_cache_path: str = "atlas_cache.npz"


@dataclass
class ContactMapConfig:
    """Configuration for Stage 1: RG Matrix Field Theory Contact Map."""

    block_size: int = 300
    t_coupling: float = 0.3
    u_coupling: float = 0.1
    newton_max_iter: int = 100
    newton_tol: float = 1e-8
    correlation_length: float = 15.0
    frg_use_threshold: int = 1000


@dataclass
class TropicalConfig:
    """Configuration for Stage 2: Hierarchical Tropical Geometry."""

    max_basins: int = 20
    block_size: int = 300
    weight_bp: float = -2.0
    weight_stack: float = -1.5


@dataclass
class RetrievalConfig:
    """Configuration for Stage 3: Topological Template Retrieval."""

    lambda_genus: float = 0.6
    lambda_wasserstein: float = 0.4
    sgw_n_projections: int = 200
    prefilter_top_k: int = 100
    retrieval_top_k: int = 10


@dataclass
class RiemannianConfig:
    """Configuration for Stage 4: Riemannian Torsion Refinement."""

    n_steps: int = 500
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    block_size: int = 50
    n_workers: int = 8
    use_symplectic: bool = True
    symplectic_h: float = 0.005
    tda_check_interval: int = 50


@dataclass
class TDAConfig:
    """Configuration for Stage 6: TDA Verification Feedback Loop."""

    wasserstein_epsilon: float = 0.5
    max_retries: int = 10
    geodesic_kick_scale: float = 0.1


@dataclass
class DomainConfig:
    """Configuration for Stage 7: Spectral Domain Decomposition."""

    use_threshold_length: int = 500
    min_domain_size: int = 30
    max_domain_size: int = 400
    se3_lr: float = 0.001
    se3_steps: int = 200


@dataclass
class PipelineConfig:
    """Master configuration for the entire TOPOMATRIX-RNA pipeline."""

    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    contact: ContactMapConfig = field(default_factory=ContactMapConfig)
    tropical: TropicalConfig = field(default_factory=TropicalConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    riemannian: RiemannianConfig = field(default_factory=RiemannianConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    n_predictions: int = 5
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: str = "predictions/"
    train_sequences: str = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
    train_labels: str = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
    validation_sequences: str = "/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv"
    validation_labels: str = "/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
    test_sequences: str = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
    sample_submission: str = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
