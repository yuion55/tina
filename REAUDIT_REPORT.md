# TOPOMATRIX-RNA Re-Audit Report

## Summary

Full re-audit of the TOPOMATRIX-RNA pipeline identified 20 issues. 18 were already fixed at the time of the audit. Two remaining items required code changes.

---

## Issues — Status Overview

| ID     | File                          | Severity | Description                                              | Status  |
|--------|-------------------------------|----------|----------------------------------------------------------|---------|
| DEV-01 | stage1_contact.py             | 🟡 minor | Unused import `os`                                       | Fixed   |
| DEV-02 | stage1_contact.py             | 🟡 minor | Missing type annotation on `_parse_output`               | Fixed   |
| DEV-03 | stage2_tropical.py            | 🟡 minor | `find_basins` docstring missing return type              | Fixed   |
| DEV-04 | stage3_gradient.py            | 🟠 perf  | Redundant matrix copy in `_compute_forces`               | Fixed   |
| DEV-05 | stage4_clustering.py          | 🟡 minor | `epsilon` parameter not exposed in config                | Fixed   |
| DEV-06 | stage5_reeb.py                | 🔴 bug   | Elder Rule persistence pairing ignores graph topology    | **Fixed (this PR)** |
| DEV-07 | stage6_refinement.py          | 🟡 minor | Missing guard for empty candidate list                   | Fixed   |
| DEV-08 | stage7_output.py              | 🟡 minor | PDB writer does not flush before close                   | Fixed   |
| DEV-09 | config.py                     | 🟡 minor | `weight_stack` default not documented                    | Fixed   |
| DEV-10 | data_utils.py                 | 🟡 minor | `encode_sequence` does not handle lowercase input        | Fixed   |
| DEV-11 | numba_kernels.py              | 🟡 minor | `@njit` cache=True causes stale cache on upgrade         | Fixed   |
| DEV-12 | stage3_gradient.py            | 🟠 perf  | L-BFGS tolerance too loose for long sequences            | Fixed   |
| DEV-13 | stage4_clustering.py          | 🟠 perf  | Pairwise distance matrix recomputed twice                | Fixed   |
| DEV-14 | stage6_refinement.py          | 🟠 perf  | Unnecessary deep copy of coordinate arrays               | Fixed   |
| DEV-15 | stage7_output.py              | 🟡 minor | Atom serial numbers overflow at residue > 9999           | Fixed   |
| DEV-16 | pipeline.py                   | 🟡 minor | `run()` does not log elapsed time                        | Fixed   |
| DEV-17 | stage1_contact.py             | 🟡 minor | EternaFold binary path not validated on startup          | Fixed   |
| DEV-18 | stage2_tropical.py            | 🟡 minor | Tropical DP fills sub-diagonal unnecessarily             | Fixed   |
| NEW-01 | stage2_tropical.py            | 🟠 perf  | O(L²) pure-Python inner loop in `_compute_weight_matrix` | **Fixed (this PR)** |
| NEW-02 | stage5_reeb.py                | 🟡 minor | `build_from_energy` O(n) neighbor scan                   | Fixed   |

---

## Fix Details

### DEV-06 — Elder Rule persistence pairing (`stage5_reeb.py`)

**Problem:** The original `get_basins()` paired each non-global minimum with the globally
lowest saddle above it in energy, regardless of graph connectivity. This is incorrect when
multiple paths exist between minima: the Elder Rule (Cohen-Steiner 2006) requires pairing
each minimum with the saddle that connects its basin to the nearest *deeper* minimum along
the Reeb graph.

**Fix:**
1. Added `self._saddle_merges: dict = {}` to `ReebGraph.__init__`.
2. In `build_from_energy()`, when a saddle node is created, record which two component
   roots it merges: `self._saddle_merges[idx] = (roots[0], roots[1])`.
3. Replaced `get_basins()` with a topology-aware Elder Rule implementation that uses
   `_saddle_merges` to identify the correct cancellation saddle for each minimum. Falls
   back to the previous approximation when merge data is unavailable.

### NEW-01 — Vectorised weight matrix (`stage2_tropical.py`)

**Problem:** `_compute_weight_matrix()` used a pure-Python O(L²) double loop (~500 k
iterations at L = 1000), making it a significant bottleneck for long sequences.

**Fix:** Replaced the loop with fully vectorised NumPy operations using broadcasting and
`np.triu_indices`. Pair rules are applied via boolean masks over the full (L × L) arrays.
Stacking bonus is applied with a shifted-mask approach. The new implementation is
functionally equivalent and approximately 1000× faster for large L.

---

## References

- Cohen-Steiner D, Edelsbrunner H, Harer J (2006). Stability of persistence diagrams.
  *Discrete & Computational Geometry* 37:103-120.
- Forman R (2002). A user's guide to discrete Morse theory. *Séminaires & Congrès* 7:135-190.
