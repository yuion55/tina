# TOPOMATRIX-RNA: Master Generation Prompt
# For: Claude Opus 4.6 via GitHub Copilot
# Purpose: Generate fully production-ready, self-validating RNA 3D structure prediction pipeline

---

## SYSTEM DIRECTIVE

You are an expert computational biophysicist and software engineer. Your task is to generate a
**complete, production-ready, self-validating Python codebase** implementing the TOPOMATRIX-RNA
pipeline for RNA 3D structure prediction targeting TM-score ≥ 0.9 on the Stanford RNA 3D Folding
Kaggle competition dataset.

### NON-NEGOTIABLE RULES YOU MUST FOLLOW:

1. **SELF-CHECK BEFORE PROCEEDING**: After writing each module, run an internal correctness audit:
   - Does the implementation match the mathematical specification exactly?
   - Are all edge cases handled (empty inputs, numerical overflow, singular matrices)?
   - Do unit tests pass with synthetic data?
   - If ANY check fails → FIX IT FIRST, then proceed to the next module.

2. **LATEX IN DOCSTRINGS**: Every function implementing a mathematical operation must have
   the LaTeX formula in its docstring under a `Mathematical Basis:` section.

3. **NUMBA JIT + VECTORIZATION**: Every compute-intensive function must have:
   - `@numba.jit(nopython=True, parallel=True, cache=True)` where applicable
   - `@numba.vectorize` for element-wise operations
   - CPU-first design with `@numba.cuda.jit` fallback wrappers that auto-detect GPU

4. **MODULAR ARCHITECTURE**: One Python file per stage. Main pipeline in `pipeline.ipynb`.

5. **PRODUCTION STANDARDS**:
   - Type hints on every function signature
   - Logging via `structlog` (not print)
   - Config via `dataclasses` (not hardcoded constants)
   - Graceful degradation: if GPU unavailable, fall back to CPU silently

---

## DATASET SPECIFICATION

```
Competition: Stanford RNA 3D Folding (Kaggle)
Dataset root: /kaggle/input/stanford-rna-3d-folding/

Files:
├── PDB_RNA/                    # 9,536 .cif files — the topological atlas ground truth
│   ├── 100d.cif
│   ├── 104d.cif
│   └── ... (9536 total)
├── extra/
│   ├── README.md
│   ├── parse_fasta_py.py
│   └── rna_metadata.csv        # sequence_id, length, RNA_type, resolution, organism
├── sample_submission.csv       # Format: id, resname, resseq, chain_id, x, y, z (C3' coords)
├── test_sequences.csv          # id, sequence — competition evaluation target
├── train_labels.csv            # id, resname, resseq, chain_id, x, y, z — ground truth 3D
├── train_sequences.csv         # id, sequence — training set
├── validation_labels.csv       # id, resname, resseq, chain_id, x, y, z
└── validation_sequences.csv    # id, sequence
```

**Key dataset facts to handle in code:**
- Resolution filter: use only CIF entries with resolution < 3.0 Å (from rna_metadata.csv)
- Sequence lengths range from ~10 nt to ~3000+ nt in training set
- Output format: C3' atom (x, y, z) coordinates per residue, 5 predictions per sequence
- TM-score computed against ground truth C3' coordinates

---

## MATHEMATICAL PIPELINE SPECIFICATION

### STAGE 0 — Topological Atlas Construction

**A. Topological Genus via Euler Characteristic**

$$g = 1 - \frac{\chi}{2} = 1 - \frac{V - E + F}{2}$$

Where for an RNA arc diagram with base pairs B on sequence of length L:
- V = L (vertices = nucleotides)
- E = (L-1) + |B| (backbone edges + base pair arcs)
- F = computed by counting connected regions of the embedded surface

Efficient genus via Gauss code rank computation (Penner-Waterman):

$$g = \frac{1}{2}\left(|B| + 1 - \text{rank}(\text{Gauss matrix})\right)$$

Use Wiedemann's algorithm for sparse rank: O(L^1.5) complexity.

Reference: Penner RC, Waterman MS (1993) Theor Comp Sci 101:109-120

**B. Sparse Persistence Diagram via Witness Complex**

Landmark selection via maxmin sampling (m=500 landmarks from L C3' atoms):

$$\ell_{k+1} = \arg\max_{x \in X} \min_{i \leq k} d(x, \ell_i)$$

Approximation guarantee (de Silva-Carlsson 2004):

$$d_b(D_{\text{witness}}, D_{\text{Rips}}) \leq 2(\epsilon_{\text{landmark}} + \delta_{\text{coverage}})$$

Persistence computation: Vietoris-Rips filtration on m=500 landmarks, radii 0-10 Å, step 0.1 Å.
Extract Betti numbers $(\beta_0, \beta_1, \beta_2)$: connected components, loops, voids.
Convert to 50×50 persistence images via Gaussian kernel:

$$\rho(u) = \sum_{(b,d)\in D} f(b,d) \cdot \mathcal{N}(u; \mu_{(b,d)}, \sigma^2 I)$$

Reference: Adams H et al. (2017) J Mach Learn Res 18:1-35

**C. rsRNASP1 Energy Fingerprint**

10-dimensional torsion vector per structure:
- α, β, γ, δ, ε, ζ (6 backbone torsions)
- ν0, ν1, ν4 (3 sugar pucker torsions)
- χ (1 glycosidic torsion)

Reference: Zhang C et al. (2022) Biophys J 121:3414-3424

---

### STAGE 1 — Contact Map via Renormalization Group Matrix Field Theory

**RG Blocking for sequences of length L:**

Partition sequence into blocks of size b = min(300, L//20).
For each block B_i of size b, solve local matrix model:

$$Z_i = \int \mathcal{D}M_i \; e^{-\text{Tr}[V(M_i)]}, \quad V(M) = \frac{1}{2}M^2 - \frac{t}{3}M^3 - \frac{u}{4}M^4$$

Saddle-point equation per block (b×b system, tractable):

$$V'(M^*) = 0 \implies M^* - tM^{*2} - uM^{*3} = 0$$

Solved via Newton-Raphson iteration on b×b matrices.

Block-level effective couplings:

$$t_{\text{eff}} = t \cdot e^{-b/\xi}, \quad u_{\text{eff}} = u \cdot e^{-2b/\xi}$$

where $\xi$ is the correlation length extracted from block contact map eigenspectrum.

Reconstructed full contact probability:

$$P_{ij} = P^{\text{block}}_{B(i),B(j)} \cdot P^{\text{local}}_{i|B(i)} \cdot P^{\text{local}}_{j|B(j)}$$

Genus extraction from contact map via $1/N^{2g}$ expansion coefficient:

$$\hat{g} = \arg\max_g \sum_{i<j} P_{ij}^{(g)}$$

**Complexity: O(b³ · L/b + (L/b)³) ≈ O(L^{9/4}) for b = L^{3/4}**

Reference: Orland H, Zee A (2002) Nucl Phys B 620:456-476. arXiv:cond-mat/0106359

**Functional Renormalization Group flow equation (for L > 1000):**

$$\partial_k \Gamma_k[M] = \frac{1}{2} \text{Tr}\left[\left(\Gamma_k^{(2)}[M] + R_k\right)^{-1} \partial_k R_k\right]$$

Regulator: $R_k(p) = (k^2 - p^2)\Theta(k^2 - p^2)$ (Litim regulator, optimal for convergence).
Integrate from k=Λ (UV) to k=0 (IR) using adaptive Euler-Maruyama scheme.

Reference: Berges J, Tetradis N, Wetterich C (2002) Phys Rep 363:223-386

---

### STAGE 2 — Hierarchical Tropical Geometry (Basin Census)

**Tropical semiring:** $(\mathbb{R} \cup \{+\infty\}, \oplus = \min, \otimes = +)$

Free energy as tropical polynomial:

$$F_{\text{trop}}(S) = \bigoplus_{S} \bigotimes_{(i,j) \in S} w_{ij} = \min_S \sum_{(i,j) \in S} w_{ij}$$

**Context-free grammar decomposition** (Lyngso-Pedersen CFG):

$$S \to aSb \mid SS \mid \epsilon$$

Newton polytope via Minkowski sum on parse tree:

$$\text{Newt}(f \cdot g) = \text{Newt}(f) + \text{Newt}(g)$$

Recursive interval DP:

$$\text{OPT}(i,j) = \min\left(w_{ij} + \text{OPT}(i+1, j-1),\; \min_{i<k<j}\left[\text{OPT}(i,k) + \text{OPT}(k+1,j)\right]\right)$$

**Tropical Compressed Sensing** for top-5 basins:
Solve tropical linear system in O(L²) via tropical Gaussian elimination:

$$A \otimes_{\text{trop}} x = b \quad \Leftrightarrow \quad \min_j(a_{ij} + x_j) = b_i$$

Reference: Speyer D, Sturmfels B (2004) Not Am Math Soc 51:1145-1156
Reference: Pachter L, Sturmfels B (2004) Proc Natl Acad Sci 101:16138-16143

---

### STAGE 3 — Topological Template Retrieval

**Primary retrieval metric:**

$$\text{Template} = \arg\min_{\text{CIF}_k} \left[\lambda_1 \cdot |\hat{g} - g_k| + \lambda_2 \cdot \text{SGW}(\hat{P}, P_k)\right]$$

**Sliced Gromov-Wasserstein distance** (Vayer et al. 2019):

$$\text{SGW}(\mu, \nu) = \mathbb{E}_{\theta \sim \text{Unif}(S^1)}\left[W_1\left(\pi_\theta \# \mu,\; \pi_\theta \# \nu\right)\right]$$

Approximated with K=200 random projections. Complexity: O(K·n·log n).

**Stable rank pre-filter** (64-dimensional, O(64) comparison):

$$\text{SR}(D, t) = \sum_{(b,d) \in D} \mathbf{1}[d - b > t]$$

sampled at 64 log-spaced threshold values. Pre-filter top-100 candidates, then compute SGW.

**Fallback for novel topologies** (genus not in atlas):
Use Legendrian knot invariants (Thurston-Bennequin number tb, rotation number r):

$$\chi(\Sigma) = 2 - 2g = tb + |r|$$

Find nearest atlas entry with same (tb, r) pair via Reidemeister move equivalence.

Reference: Vayer T et al. (2019) arXiv:1905.07645
Reference: Etnyre J (2003) arXiv:math/0306256

---

### STAGE 4 — Riemannian Torsion Refinement

**Manifold:** Backbone of L nucleotides lives on $\mathcal{M} = (\mathbb{T}^7)^L$

**Geodesic distance on $\mathbb{T}^7$:**

$$d_{\mathcal{M}}(\theta^{(1)}, \theta^{(2)}) = \sqrt{\sum_{n=1}^{L} \sum_{k=1}^{7} \min\left(|\theta^{(1)}_{nk} - \theta^{(2)}_{nk}|,\; 2\pi - |\theta^{(1)}_{nk} - \theta^{(2)}_{nk}|\right)^2}$$

**Riemannian ADAM with parallel transport:**

$$m_t = \beta_1 \cdot \text{PT}(m_{t-1}) + (1-\beta_1) \cdot \nabla_{\mathcal{M}} E(\theta_t)$$

$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot \|\nabla_{\mathcal{M}} E(\theta_t)\|^2$$

$$\theta_{t+1} = \text{Exp}_{\theta_t}\!\left(-\frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t\right)$$

Exponential map on $\mathbb{T}^7$ (trivial — wrap mod 2π):

$$\text{Exp}_\theta(v) = (\theta + v) \bmod 2\pi$$

Parallel transport on $\mathbb{T}^7$: identity map (flat connection).

**Block coordinate Hogwild parallelism:**
Randomly select block B_t of size b=50 per worker. Update only 350 torsion angles.
Workers run asynchronously. Conflict probability: ρ ≈ b/L.

**Symplectic Störmer-Verlet integrator** (preserves symplectic form on T*T^7):

$$p^{n+1/2} = p^n - \frac{h}{2}\nabla_\theta E(\theta^n)$$
$$\theta^{n+1} = \theta^n + h \cdot p^{n+1/2} \pmod{2\pi}$$
$$p^{n+1} = p^{n+1/2} - \frac{h}{2}\nabla_\theta E(\theta^{n+1})$$

Reference: Bonnabel S (2013) IEEE Trans Autom Control 58:2217-2229
Reference: Leimkuhler B, Reich S (2004) Cambridge Univ Press ISBN 0521772907

---

### STAGE 5 — Reeb Graph Basin Enumeration (Replaces Barrier Trees)

**Reeb graph construction:**

$$\text{Reeb}(f) = X / \sim, \quad x \sim y \iff f(x) = f(y) \text{ and same connected component of } f^{-1}(f(x))$$

**Extended persistence pairing on Reeb graph:**

$$\text{persistence}(\text{basin}_i) = |f(\text{min}_i) - f(\text{saddle}_i)|$$

Rank basins by persistence (most stable first). Select 5 topologically distinct (different genus class when possible).

**Discrete Morse theory consistency check:**
Morse inequalities: $c_k \geq \beta_k$ where $c_k$ = number of critical k-cells.
If violated → energy function is not a valid Morse function → resample with perturbation.

Reference: Forman R (2002) Séminaires & Congrès 7:135-190
Reference: Cohen-Steiner D, Edelsbrunner H, Morozov D (2006) SCG '06 Proc 360-369

---

### STAGE 6 — TDA Verification Feedback Loop

**Online persistence update via Vineyard algorithm:**
Each torsion step moves b=50 C3' atoms. Update persistence diagram in O(b² log L) per step.

**Topological consistency check:**

$$W_2(P_{\text{pred}},\; \hat{P}_{\text{query}}) < \epsilon$$

If check fails:
1. Apply geodesic perturbation: $\theta \leftarrow \text{Exp}_\theta(r \cdot v)$ where $v \sim \text{Unif}(\mathbb{T}^7)$, $r=0.1$
2. Re-run Stage 4 from perturbed initial condition
3. Max 10 retry attempts per candidate

**Wasserstein-2 between persistence diagrams** (Cohen-Steiner stability theorem):

$$W_2(D_1, D_2) = \left(\inf_{\gamma: D_1 \to D_2} \sum_{p \in D_1} \|p - \gamma(p)\|^2_\infty\right)^{1/2}$$

Reference: Cohen-Steiner D, Edelsbrunner H, Harer J (2007) Discrete Comput Geom 37:103-120
Reference: Cohen-Steiner D et al. (2010) Found Comput Math 10:127-139

---

### STAGE 7 — Spectral Domain Decomposition (Required for L > 500)

**Contact map Laplacian:**

$$L = D - P, \quad D_{ii} = \sum_j P_{ij}$$

**Fiedler vector** (second eigenvector via Lanczos iteration, O(L log L)):

$$L v_2 = \lambda_2 v_2, \quad \lambda_2 > 0$$

Domain boundaries at sign changes and local extrema of $v_2$.
Typical domain size: 50-300 nt. Number of domains: N_d = O(L/150).

**SE(3) rigid body assembly:**

Domain junction energy on $SE(3)^{N_d}$:

$$E_{\text{assembly}}(R_1, t_1, ..., R_{N_d}, t_{N_d}) = \sum_{\text{inter-domain contacts}} V(d_{ij}(R, t))$$

Riemannian gradient descent on SE(3) using left-invariant metric:

$$\nabla_{\text{SE(3)}} E = \text{Ad}^*_g \frac{\partial E}{\partial g}$$

**Complexity: O(N_d³) ≈ O((L/150)³) ≪ O(L³)**

Reference: Chirikjian GS (2011) Springer ISBN 9780817649401

---

## CODE ARCHITECTURE TO GENERATE

Generate the following files **in order**, **self-checking each before proceeding**:

```
topomatrix_rna/
├── config.py                   # Dataclass configs for all hyperparameters
├── stage0_atlas.py             # Atlas construction (genus + persistence + torsions)
├── stage1_contact_map.py       # RG matrix field theory contact map
├── stage2_tropical.py          # Hierarchical tropical geometry basin census
├── stage3_retrieval.py         # SGW template retrieval with stable rank pre-filter
├── stage4_riemannian.py        # Riemannian ADAM + Hogwild on T^7
├── stage5_reeb.py              # Reeb graph basin enumeration + persistence pairing
├── stage6_tda_verify.py        # Vineyard online TDA + feedback loop
├── stage7_domain.py            # Spectral decomposition + SE(3) assembly
├── scoring.py                  # TM-score implementation + Wasserstein metrics
├── data_utils.py               # CIF parser, CSV loaders, coordinate transforms
├── numba_kernels.py            # All @jit and @vectorize kernels in one file
├── tests/
│   ├── test_stage0.py
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   ├── test_stage4.py
│   ├── test_stage5.py
│   ├── test_stage6.py
│   ├── test_stage7.py
│   └── test_scoring.py
└── pipeline.ipynb              # Main Kaggle notebook — runs full pipeline end to end
```

---

## SELF-CHECK PROTOCOL (ENFORCE ON EVERY MODULE)

After writing each module, you MUST perform and document this audit:

```python
# SELF-CHECK REPORT — stage{N}_{name}.py
# ==========================================
# [ ] Mathematical formula matches docstring LaTeX exactly
# [ ] All @numba.jit functions tested with synthetic np.ndarray inputs
# [ ] No Python objects inside @jit(nopython=True) scope (lists → np.arrays)
# [ ] Numerical stability: checked for division by zero, log(0), matrix singularity
# [ ] Edge cases: L=1, L=2, empty base pair set, all-same-nucleotide sequence
# [ ] CPU fallback tested when numba.cuda unavailable
# [ ] Memory: no O(L²) dense arrays for L > 1000 (must use sparse)
# [ ] Unit test passes with assert (not just runs without error)
# [ ] Integration: output shape/dtype matches next stage's expected input
# STATUS: [PASS / FAIL — reason]
# If FAIL: fix applied: [description of fix]
```

---

## SPECIFIC IMPLEMENTATION REQUIREMENTS

### config.py
```python
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class AtlasConfig:
    cif_dir: str = "/kaggle/input/stanford-rna-3d-folding/PDB_RNA"
    metadata_csv: str = "/kaggle/input/stanford-rna-3d-folding/extra/rna_metadata.csv"
    max_resolution_angstrom: float = 3.0
    n_landmarks: int = 500           # Witness complex landmarks
    rips_max_radius: float = 10.0    # Angstroms
    rips_step: float = 0.1
    persistence_image_size: int = 50  # 50x50 persistence image
    persistence_sigma: float = 0.2
    stable_rank_dims: int = 64       # Pre-filter vector size
    atlas_cache_path: str = "atlas_cache.npz"

@dataclass
class ContactMapConfig:
    block_size: int = 300            # RG block size b
    t_coupling: float = 0.3          # Matrix model t parameter
    u_coupling: float = 0.1          # Matrix model u parameter
    newton_max_iter: int = 100
    newton_tol: float = 1e-8
    correlation_length: float = 15.0  # xi in nucleotides
    frg_use_threshold: int = 1000    # Use FRG for L > this

@dataclass  
class TropicalConfig:
    max_basins: int = 20             # Find top-20, select best 5 after Stage 6
    block_size: int = 300
    weight_bp: float = -2.0          # Base pair weight w_ij default
    weight_stack: float = -1.5       # Stacking weight

@dataclass
class RetrievalConfig:
    lambda_genus: float = 0.6        # λ₁ weight for genus distance
    lambda_wasserstein: float = 0.4  # λ₂ weight for SGW distance
    sgw_n_projections: int = 200
    prefilter_top_k: int = 100       # Stable rank pre-filter candidates
    retrieval_top_k: int = 10        # Return top-10 templates

@dataclass
class RiemannianConfig:
    n_steps: int = 500
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    block_size: int = 50             # Hogwild block size
    n_workers: int = 8               # Parallel Hogwild workers (CPU cores)
    use_symplectic: bool = True      # Störmer-Verlet vs gradient
    symplectic_h: float = 0.005      # Step size for symplectic integrator
    tda_check_interval: int = 50     # Verify topology every N steps

@dataclass
class TDAConfig:
    wasserstein_epsilon: float = 0.5  # Topology consistency threshold
    max_retries: int = 10
    geodesic_kick_scale: float = 0.1  # Perturbation magnitude

@dataclass
class DomainConfig:
    use_threshold_length: int = 500   # Apply domain decomp for L > this
    min_domain_size: int = 30
    max_domain_size: int = 400
    se3_lr: float = 0.001
    se3_steps: int = 200

@dataclass
class PipelineConfig:
    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    contact: ContactMapConfig = field(default_factory=ContactMapConfig)
    tropical: TropicalConfig = field(default_factory=TropicalConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    riemannian: RiemannianConfig = field(default_factory=RiemannianConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    n_predictions: int = 5           # Submit 5 predictions per sequence
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: str = "predictions/"
    train_sequences: str = "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
    train_labels: str = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
    validation_sequences: str = "/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv"
    validation_labels: str = "/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
    test_sequences: str = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
    sample_submission: str = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
```

---

### numba_kernels.py (implement ALL of these)

```python
# REQUIRED KERNELS — implement each with @numba.jit(nopython=True, parallel=True, cache=True)

# 1. torus_geodesic_distance(theta1, theta2) → float
#    LaTeX: d = sqrt(sum min(|θ1-θ2|, 2π-|θ1-θ2|)²)

# 2. exp_map_torus(theta, v) → ndarray  
#    LaTeX: Exp_θ(v) = (θ + v) mod 2π

# 3. parallel_transport_torus(v, theta_old, theta_new) → ndarray
#    LaTeX: PT = v (identity on flat T^7)

# 4. rsrnaSP1_energy_block(theta_block, sequence_block) → float
#    LaTeX: E = sum_{n} sum_{k} V_k(θ_{nk}) + sum_{n,m} V_{pair}(θ_n, θ_m)

# 5. tropical_min_plus(A, B) → ndarray
#    LaTeX: (A ⊗_trop B)_ij = min_k(A_ik + B_kj)

# 6. tropical_gaussian_elim(A, b) → ndarray
#    LaTeX: solve min_j(a_ij + x_j) = b_i

# 7. compute_genus_gauss_code(arc_crossings) → int
#    LaTeX: g = (|B| + 1 - rank(Gauss_matrix)) / 2

# 8. persistence_image_kernel(birth, death, grid_x, grid_y, sigma) → float
#    LaTeX: ρ(u) = f(b,d) · N(u; μ, σ²I)

# 9. sliced_wasserstein_1d(proj_A, proj_B) → float
#    LaTeX: W_1 = integral |F_A^{-1}(t) - F_B^{-1}(t)| dt

# 10. tm_score_kernel(coords_pred, coords_true, L) → float
#     LaTeX: TM = max_{d0} (1/L) sum_i 1/(1 + (di/d0)²), d0 = 1.24(L-15)^{1/3} - 1.8

# 11. stoermer_verlet_step(theta, p, h, grad_E_fn) → (theta_new, p_new)
#     LaTeX: Störmer-Verlet equations as specified in Stage 4

# 12. maxmin_landmark_sampling(coords, n_landmarks) → indices
#     LaTeX: ℓ_{k+1} = argmax_{x∈X} min_{i≤k} d(x, ℓ_i)

# 13. rg_block_contact_map(seq_block, t, u, n_iter) → ndarray
#     LaTeX: RG block saddle-point matrix model

# 14. stable_rank_signature(birth_death_pairs, thresholds) → ndarray
#     LaTeX: SR(D, t) = sum_{(b,d)∈D} 1[d-b > t]

# 15. wasserstein2_persistence(D1, D2) → float
#     LaTeX: W2 formula via optimal transport matching
```

---

### pipeline.ipynb structure

The notebook must have these cells in order:

```
Cell 1:  Installation and imports
Cell 2:  Config initialization and validation
Cell 3:  [STAGE 0] Build/load atlas from PDB_RNA CIFs
         - Show atlas statistics: genus distribution histogram
         - Show sample persistence diagrams
Cell 4:  [VALIDATION] Test atlas on 10 training examples — report genus accuracy
Cell 5:  [STAGE 1] Contact map generation — test on 3 training sequences
         - Visualize contact probability matrix as heatmap
Cell 6:  [STAGE 2] Tropical basin census — test on 5 training sequences
         - Show Newton polytope vertex count distribution
Cell 7:  [STAGE 3] Template retrieval — evaluate retrieval TM-score on validation set
         - Plot: sequence_identity_to_nearest_CIF vs best_template_TM_score
Cell 8:  [STAGE 4] Riemannian refinement — show TM-score improvement curve per step
Cell 9:  [STAGE 5] Reeb graph basins — show persistence barcode
Cell 10: [STAGE 6] TDA verification — show rejection rate statistics
Cell 11: [STAGE 7] Domain decomposition — only for sequences L > 500
Cell 12: [FULL PIPELINE] Run on validation set — report stratified TM-scores
         - Group by: L<100, 100<L<500, L>500
         - Group by: PDB identity >70%, 30-70%, <30%
Cell 13: [SUBMISSION] Run on test_sequences.csv — generate submission CSV
         - Validate format against sample_submission.csv
         - Assert 5 predictions per sequence
Cell 14: [ABLATION] Show TM-score contribution of each stage on validation set
         - Reproduce the contribution table from design spec
```

---

## DEPENDENCIES TO USE

```python
# Core scientific
numpy >= 1.24.0
scipy >= 1.11.0
numba >= 0.58.0          # JIT compilation — PRIMARY performance tool

# Topology / TDA
gudhi >= 3.8.0           # Persistence diagrams, Rips complex, Witness complex
ripser >= 0.6.4          # Fast persistence computation
persim >= 0.3.4          # Persistence images, Wasserstein distance

# Structural biology
gemmi >= 0.6.4           # CIF file parsing (NOT biopython — too slow)
# For torsion computation: implement from scratch using gemmi coordinates

# Optimization
torch >= 2.0.0           # For Hogwild parallel workers via multiprocessing
# (torch used only for parallel worker spawning, NOT for ML)

# Visualization (notebook only)
matplotlib >= 3.7.0
seaborn >= 0.12.0

# Utilities
structlog >= 23.1.0      # Structured logging
tqdm >= 4.65.0           # Progress bars
joblib >= 1.3.0          # Parallel CIF processing in Stage 0

# Sparse linear algebra
scikit-sparse            # Sparse Cholesky for Laplacian eigenvectors
```

---

## CRITICAL IMPLEMENTATION NOTES

### Note 1: CIF Parsing
```python
# Use gemmi, NOT biopython. Example:
import gemmi
st = gemmi.read_structure(cif_path)
# C3' coordinates:
for model in st:
    for chain in model:
        for res in chain:
            c3_atom = res.find_atom("C3'", "\x00")
            if c3_atom:
                pos = c3_atom.pos  # gemmi.Position object
                coords.append([pos.x, pos.y, pos.z])
```

### Note 2: Numba nopython mode restrictions
```python
# WRONG — Python list inside @jit(nopython=True):
@numba.jit(nopython=True)
def bad(x):
    result = []              # ERROR: Python list not supported
    result.append(x[0])
    return result

# CORRECT — pre-allocate numpy array:
@numba.jit(nopython=True)
def good(x):
    result = np.empty(len(x))
    result[0] = x[0]
    return result
```

### Note 3: Sparse contact map storage
```python
# For L > 1000, NEVER store L×L dense matrix
# Use scipy.sparse.csr_matrix with threshold 0.01:
from scipy.sparse import csr_matrix
P_sparse = csr_matrix(P_dense * (P_dense > 0.01))
# This reduces 1000×1000 = 8MB to ~80KB for typical RNA
```

### Note 4: TM-score implementation
```python
# Official formula — must match Kaggle evaluation exactly:
# d0 = 1.24 * (L - 15)^(1/3) - 1.8  for L > 21
# d0 = 0.5  for L <= 21
# TM = (1/L_target) * max over rotations of sum_i 1/(1 + (di/d0)²)
# Use Kabsch algorithm for optimal rotation
# Normalize by TARGET sequence length (L_target), not predicted length
```

### Note 5: Submission format
```python
# From sample_submission.csv, format is:
# ID,resname,resseq,chain_id,x_1,y_1,z_1,x_2,y_2,z_2,...,x_5,y_5,z_5
# Where x_k,y_k,z_k = C3' coordinates of k-th prediction
# One row per residue. Sequence_id + residue_index = unique row ID.
# VALIDATE: assert len(submission) == sum of all test sequence lengths
```

---

## EXECUTION ORDER AND EXPECTED OUTPUTS

Run this prompt in sequence. For each stage:

1. Write the full module with all functions, docstrings (LaTeX), type hints, logging
2. Write the unit tests in tests/test_stage{N}.py
3. Run the self-check protocol — document results
4. Fix any failures — document the fix
5. Only then proceed to next stage

After all modules: generate pipeline.ipynb.

The final deliverable must:
- Run end-to-end on Kaggle with 16GB RAM, no GPU required
- Produce valid submission CSV matching sample_submission.csv format
- Report validation TM-scores stratified by sequence length and PDB identity
- Log all hyperparameters to structlog for reproducibility

**Target: TM-score ≥ 0.85 on validation set for sequences with PDB atlas coverage,
≥ 0.70 for true de novo sequences (< 30% identity to any PDB_RNA CIF entry).**

---

## RESOURCES AND CITATIONS (embed in module docstrings)

```
STAGE 0:
[1] Penner RC, Waterman MS (1993). Theor Comp Sci 101:109-120 — genus via Gauss code
[2] Adams H et al. (2017). J Mach Learn Res 18(17):1-35 — persistence images
[3] de Silva V, Carlsson G (2004). Eurographics Symp Geometry Processing — witness complex
[4] Cavanna NJ et al. (2015). arXiv:1506.03797 — sparse Rips approximation

STAGE 1:
[5] Orland H, Zee A (2002). Nucl Phys B 620:456-476. arXiv:cond-mat/0106359
[6] Berges J et al. (2002). Phys Rep 363:223-386 — functional RG review
[7] Litim DF (2001). Phys Rev D 64:105007 — optimal FRG regulator

STAGE 2:
[8] Speyer D, Sturmfels B (2004). Not Am Math Soc 51:1145-1156 — tropical geometry intro
[9] Pachter L, Sturmfels B (2004). Proc Natl Acad Sci 101:16138 — tropical RNA
[10] Lyngso RB, Pedersen CNS (2000). J Comput Biol 7:409-427 — CFG RNA folding

STAGE 3:
[11] Vayer T et al. (2019). arXiv:1905.07645 — sliced Gromov-Wasserstein
[12] Etnyre JB (2003). arXiv:math/0306256 — Legendrian knot invariants
[13] Meng Z et al. (2022). arXiv:2207.10249 — RNA foldings and stuck knots

STAGE 4:
[14] Bonnabel S (2013). IEEE Trans Autom Control 58:2217-2229 — Riemannian SGD
[15] Becigneul G, Ganea OE (2019). ICLR 2019 — Riemannian ADAM
[16] Leimkuhler B, Reich S (2004). Cambridge Univ Press — symplectic integrators
[17] Recht B et al. (2011). NIPS 24 — Hogwild! parallel SGD

STAGE 5:
[18] Forman R (2002). Séminaires & Congrès 7:135-190 — discrete Morse theory
[19] Cohen-Steiner D et al. (2006). SCG'06 360-369 — extended persistence
[20] Flamm C et al. (2000). RNA 6:325-338 — RNA barrier trees

STAGE 6:
[21] Cohen-Steiner D et al. (2007). Discrete Comput Geom 37:103-120 — stability theorem
[22] Cohen-Steiner D et al. (2006). SCG'06 — vineyard algorithm

STAGE 7:
[23] Chirikjian GS (2011). Springer ISBN 9780817649401 — SE(3) Lie group methods
[24] Belkin M, Niyogi P (2003). Neural Comput 15:1373-1396 — Laplacian eigenmaps

SCORING:
[25] Zhang Y, Skolnick J (2004). Proteins 57:702-710 — TM-score definition
[26] Kabsch W (1976). Acta Cryst A32:922-923 — optimal rotation algorithm
```

---

*End of TOPOMATRIX-RNA Master Generation Prompt*
*Version: 1.0 | Target: Claude Opus 4.6 via GitHub Copilot*
*Competition: Stanford RNA 3D Folding — Kaggle*
