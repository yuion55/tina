# Tina

## TOPOMATRIX-RNA: Topological Matrix Field Theory Pipeline for RNA 3D Structure Prediction

This repository implements the TOPOMATRIX-RNA pipeline, a production-ready, self-validating RNA 3D structure prediction system targeting TM-score ≥ 0.9 on the Stanford RNA 3D Folding Kaggle competition dataset. The pipeline combines topological data analysis (TDA), renormalization group matrix field theory, tropical geometry, and Riemannian optimization to predict RNA tertiary structures from sequence alone.

### Key Features

- **High Accuracy**: Targets TM-score ≥ 0.9, surpassing current state-of-the-art methods
- **Scalable**: Handles RNA sequences from ~10 to 3000+ nucleotides
- **Modular Design**: 8-stage pipeline with independent, testable modules
- **Production Ready**: Full logging, configuration management, and error handling
- **GPU Accelerated**: Numba JIT compilation with CUDA fallbacks
- **Self-Validating**: Built-in topological verification and feedback loops

### Pipeline Overview

The TOPOMATRIX-RNA pipeline consists of 8 stages:

1. **Stage 0: Topological Atlas Construction**
   - Builds a persistent homology atlas from high-resolution RNA structures
   - Uses Gudhi/ripser for Vietoris-Rips complexes
   - Generates persistence images and stable rank features

2. **Stage 1: Contact Map via RG Matrix Field Theory**
   - Implements renormalization group flow for matrix field theory
   - Predicts base-base contact probabilities using Boltzmann weights
   - Uses Newton-Raphson optimization for fixed points

3. **Stage 2: Hierarchical Tropical Geometry (Basin Census)**
   - Applies tropical semiring algebra for hierarchical structure enumeration
   - Census of energy basins using min-plus algebra
   - Block-wise computation for memory efficiency

4. **Stage 3: Topological Template Retrieval**
   - Retrieves structurally similar templates using sliced Wasserstein distances
   - Combines genus filtration and persistence-based matching
   - Pre-filters candidates for computational efficiency

5. **Stage 4: Riemannian Torsion Refinement**
   - Refines structures using symplectic Riemannian optimization
   - Incorporates RNA-specific torsion angle constraints
   - Uses Adam optimizer with symplectic integrators

6. **Stage 5: Reeb Graph Basin Enumeration**
   - Enumerates conformational basins via Reeb graphs
   - Morse theory-based landscape analysis
   - Identifies metastable states

7. **Stage 6: TDA Verification Feedback Loop**
   - Validates predictions using persistent homology
   - Computes bottleneck distances for topological consistency
   - Feedback loop for iterative refinement

8. **Stage 7: Spectral Domain Decomposition (L > 500)**
   - Decomposes long RNAs into spectral domains
   - SE(3) rigid body assembly
   - Handles tertiary interactions across domains

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yuion55/tina.git
cd tina
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional topological data analysis libraries:
```bash
pip install gudhi ripser persim
```

### Usage

#### Running the Full Pipeline

Execute the Jupyter notebook for interactive usage:
```bash
jupyter notebook topomatrix_rna/pipeline.ipynb
```

Or use the Python script for batch processing:
```python
from topomatrix_rna.pipeline_v2 import run_pipeline
from topomatrix_rna.config import PipelineConfig

config = PipelineConfig()
predictions = run_pipeline(config)
```

#### Training

Train the neural network components:
```python
from topomatrix_rna.train import train_model
from topomatrix_rna.config import TrainingConfig

config = TrainingConfig()
train_model(config)
```

#### Evaluation

Evaluate predictions against ground truth:
```python
from topomatrix_rna.scoring import evaluate_predictions

results = evaluate_predictions(predictions, ground_truth)
print(f"Mean TM-score: {results['_aggregate']['mean_tm_score']:.3f}")
```

### Dependencies

Core dependencies (requirements.txt):
- numpy >= 1.24
- scipy >= 1.10  
- numba >= 0.57
- structlog >= 23.0
- torch >= 2.6
- onnxruntime >= 1.16
- tqdm >= 4.60
- gemmi >= 0.6

Additional TDA libraries:
- gudhi
- ripser
- persim

### Data Format

The pipeline expects data in the Stanford RNA 3D Folding format:
- Sequences: CSV with columns `id`, `sequence`
- Labels: CSV with columns `id`, `resname`, `resseq`, `chain_id`, `x`, `y`, `z` (C3' coordinates)
- CIF files: For atlas construction (resolution < 3.0 Å filtered)

### Configuration

All parameters are configurable via dataclasses in `topomatrix_rna/config.py`:

- `PipelineConfig`: Master pipeline settings
- `AtlasConfig`: Topological atlas parameters
- `ContactMapConfig`: RG matrix field theory settings
- `TropicalConfig`: Tropical geometry parameters
- `RetrievalConfig`: Template retrieval weights
- `RiemannianConfig`: Optimization hyperparameters
- `RNABiologyConstants`: RNA-specific physical constants

### Mathematical Foundations

The pipeline is grounded in rigorous mathematics:

- **Persistent Homology**: Captures topological features at multiple scales
- **Renormalization Group**: Coarse-graining for contact map prediction  
- **Tropical Geometry**: Algebraic structure for basin enumeration
- **Riemannian Optimization**: Manifold-constrained structure refinement
- **Sliced Wasserstein Distance**: Topological similarity metric

### References

1. Penner RC, Waterman MS (1993). Theor Comp Sci 101:109-120
2. Orland H, Zee A (2002). Nucl Phys B 620:456-476  
3. Adams H et al. (2017). J Mach Learn Res 18(17):1-35
4. Zhang Y, Skolnick J (2004). Proteins 57:702-710
5. Saenger W (1984). Principles of Nucleic Acid Structure
6. Olson WK et al. (2001). J Mol Biol 313:229-237

### License

This project is released under the MIT License.

### Contributing

Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests for code improvements.

### Citation

If you use TOPOMATRIX-RNA in your research, please cite:

```
@software{topomatrix_rna_2024,
  title={TOPOMATRIX-RNA: Topological Matrix Field Theory for RNA Structure Prediction},
  author={Yuion55},
  year={2024},
  url={https://github.com/yuion55/tina}
}
```