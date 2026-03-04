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
class RNABiologyConstants:
    """RNA-specific biological and chemical constants.

    Sources:
        - Base pair geometry: Saenger W (1984) Principles of Nucleic Acid Structure, Springer.
        - Stacking distances: Olson WK et al. (2001) J Mol Biol 313:229-237.
        - Mg2+ coordination: Draper DE (2004) RNA 10:335-343.
        - Sugar pucker: Altona C, Sundaralingam M (1972) JACS 94:8205-8212.
        - Backbone torsion means: Richardson JS et al. (2008) RNA 14:465-481.
        - Contact map weights: Turner DH, Mathews DH (2010) NAR 38:D209-D215.
    """

    # Base pair geometry (A-form RNA helix)
    bp_dist_wc_mean: float = 10.4       # Å, C1'-C1' distance in Watson-Crick pair
    bp_dist_wc_std: float = 0.4         # Å
    helix_rise_a_form: float = 2.81     # Å per base pair
    helix_twist_a_form: float = 32.7    # degrees per base pair
    helix_diameter: float = 18.0        # Å

    # Stacking distances
    stack_dist_c3c3: float = 3.4        # Å, mean C3'-to-C3' within helix
    stack_dist_coaxial: float = 3.5     # Å, coaxial stack interface
    stack_dist_max: float = 5.0         # Å, maximum for stacking bonus

    # Mg2+ coordination
    mg_inner_sphere_dist: float = 2.07  # Å, mean Mg2+-O direct coordination
    mg_outer_sphere_dist: float = 4.15  # Å, Mg2+...O via water
    debye_length_mg_2mm: float = 7.0    # Å, Debye screening at 2 mM Mg2+
    debye_length_na_100mm: float = 10.0 # Å, Debye screening at 100 mM NaCl

    # Sugar pucker thresholds (delta torsion angle)
    c3endo_delta_min: float = 55.0      # degrees
    c3endo_delta_max: float = 110.0     # degrees
    c2endo_delta_min: float = 120.0     # degrees
    c2endo_delta_max: float = 175.0     # degrees

    # A-form backbone torsion means (degrees) — Richardson et al. 2008
    alpha_a_form: float = -68.0
    beta_a_form: float = 178.0
    gamma_a_form: float = 54.0
    delta_a_form: float = 83.0
    epsilon_a_form: float = 212.0
    zeta_a_form: float = 289.0
    chi_a_form: float = -159.0          # anti conformation

    # Contact map weights — biologically calibrated
    weight_gc_wc: float = 1.0
    weight_au_wc: float = 0.9
    weight_gu_wobble: float = 0.7
    weight_non_canonical: float = 0.4
    weight_a_minor_bonus: float = 0.3
    weight_gnra_bonus: float = 0.5
    weight_coaxial_bonus: float = 0.6
    weight_mg_bridge: float = 0.4
    weight_pseudoknot: float = 0.85
    weight_suite_penalty: float = 0.1   # per degree Mahalanobis outlier


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
    biology: RNABiologyConstants = field(default_factory=RNABiologyConstants)
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
