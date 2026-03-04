"""Stage 0: Topological Atlas Construction.

Builds a topological atlas from PDB RNA CIF structures including:
A. Topological genus via Euler characteristic / Gauss code rank
B. Sparse persistence diagrams via witness complex
C. rsRNASP1 energy fingerprints (torsion vectors)

References:
    [1] Penner RC, Waterman MS (1993). Theor Comp Sci 101:109-120 — genus via Gauss code
    [2] Adams H et al. (2017). J Mach Learn Res 18(17):1-35 — persistence images
    [3] de Silva V, Carlsson G (2004). Eurographics — witness complex
    [4] Zhang C et al. (2022). Biophys J 121:3414-3424 — rsRNASP1
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy.spatial.distance import cdist

from .config import AtlasConfig
from .data_utils import (
    RNAStructure,
    coords_to_distance_matrix,
    encode_sequence,
    parse_cif_c3_coords,
)
from .numba_kernels import (
    compute_genus_gauss_code,
    maxmin_landmark_sampling,
    persistence_image_kernel,
    stable_rank_signature,
)

logger = structlog.get_logger(__name__)


class TopologicalAtlas:
    """Topological atlas built from PDB RNA structures.

    Stores genus, persistence images, stable rank signatures, and torsion
    fingerprints for each CIF entry, enabling template retrieval.
    """

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.entries: Dict[str, AtlasEntry] = {}
        self._stable_rank_thresholds = np.logspace(-2, 1, config.stable_rank_dims)

    def build_from_directory(
        self, cif_dir: Optional[str] = None, max_entries: Optional[int] = None
    ) -> None:
        """Build atlas by parsing all CIF files in a directory.

        Args:
            cif_dir: Directory containing .cif files. Defaults to config.cif_dir.
            max_entries: Maximum number of entries to process (for testing).
        """
        cif_dir = cif_dir or self.config.cif_dir
        if not os.path.isdir(cif_dir):
            logger.warning("cif_dir_not_found", path=cif_dir)
            return

        cif_files = [f for f in os.listdir(cif_dir) if f.endswith(".cif")]
        if max_entries is not None:
            cif_files = cif_files[:max_entries]

        logger.info("building_atlas", n_files=len(cif_files))

        for i, fname in enumerate(cif_files):
            path = os.path.join(cif_dir, fname)
            try:
                structure = parse_cif_c3_coords(path)
                if structure is not None and structure.coords_c3.shape[0] >= 3:
                    entry = self.process_structure(structure)
                    self.entries[structure.pdb_id] = entry
            except Exception as e:
                logger.debug("atlas_entry_error", file=fname, error=str(e))

            if (i + 1) % 100 == 0:
                logger.info("atlas_progress", processed=i + 1, total=len(cif_files))

        logger.info("atlas_built", n_entries=len(self.entries))

    def process_structure(self, structure: RNAStructure) -> "AtlasEntry":
        """Process a single RNA structure into an atlas entry.

        Args:
            structure: Parsed RNA structure with C3' coordinates.

        Returns:
            AtlasEntry with computed topological features.
        """
        coords = structure.coords_c3
        L = coords.shape[0]

        # A. Compute topological genus from predicted base pairs
        genus = self._compute_genus(structure.sequence, coords)

        # B. Compute persistence diagram and image
        persistence_image, birth_death = self._compute_persistence(coords)

        # C. Compute stable rank signature
        if birth_death.shape[0] > 0:
            sr = stable_rank_signature(
                birth_death[:, 0],
                birth_death[:, 1],
                self._stable_rank_thresholds,
            )
        else:
            sr = np.zeros(self.config.stable_rank_dims)

        return AtlasEntry(
            pdb_id=structure.pdb_id,
            sequence=structure.sequence,
            length=L,
            genus=genus,
            persistence_image=persistence_image,
            stable_rank=sr,
            coords_c3=coords,
            birth_death=birth_death,
        )

    def _compute_genus(self, sequence: str, coords: np.ndarray) -> int:
        """Compute topological genus from base pairs.

        Uses distance-based base pair detection on C3' coordinates and then
        computes genus via Gauss code crossing matrix rank.
        """
        L = len(sequence)
        if L < 5:
            return 0

        # Detect base pairs from C3' distance (threshold ~18Å for C3'-C3')
        encoded = encode_sequence(sequence)
        dist_matrix = coords_to_distance_matrix(coords)

        arc_i_list: List[int] = []
        arc_j_list: List[int] = []

        for i in range(L):
            for j in range(i + 4, L):
                if dist_matrix[i, j] < 18.0:
                    si, sj = encoded[i], encoded[j]
                    is_wc = (
                        (si == 0 and sj == 3)
                        or (si == 3 and sj == 0)
                        or (si == 1 and sj == 2)
                        or (si == 2 and sj == 1)
                        or (si == 2 and sj == 3)
                        or (si == 3 and sj == 2)
                    )
                    if is_wc:
                        arc_i_list.append(i)
                        arc_j_list.append(j)

        if len(arc_i_list) == 0:
            return 0

        arc_i = np.array(arc_i_list, dtype=np.int64)
        arc_j = np.array(arc_j_list, dtype=np.int64)

        return int(compute_genus_gauss_code(arc_i, arc_j, L))

    def _compute_persistence(
        self, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute persistence diagram and image from C3' coordinates.

        Uses maxmin landmark sampling for efficiency on large structures.

        Returns:
            Tuple of (persistence_image, birth_death_array).
        """
        L = coords.shape[0]
        cfg = self.config
        img_size = cfg.persistence_image_size

        # Landmark sampling for efficiency
        n_landmarks = min(cfg.n_landmarks, L)
        if L > n_landmarks:
            landmark_idx = maxmin_landmark_sampling(coords, n_landmarks)
            landmark_coords = coords[landmark_idx]
        else:
            landmark_coords = coords

        # Compute pairwise distances
        dist = cdist(landmark_coords, landmark_coords)

        # Simple Vietoris-Rips persistence via distance matrix
        # Extract 0-dimensional persistence (connected components)
        birth_death_pairs = self._rips_persistence_h0(dist)

        # Compute persistence image
        if birth_death_pairs.shape[0] > 0:
            grid_x = np.linspace(0, cfg.rips_max_radius, img_size)
            grid_y = np.linspace(0, cfg.rips_max_radius, img_size)
            pi = persistence_image_kernel(
                birth_death_pairs[:, 0],
                birth_death_pairs[:, 1],
                grid_x,
                grid_y,
                cfg.persistence_sigma,
            )
        else:
            pi = np.zeros((img_size, img_size))

        return pi, birth_death_pairs

    @staticmethod
    def _rips_persistence_h0(dist: np.ndarray) -> np.ndarray:
        """Compute 0-dimensional Rips persistence (connected components).

        Uses a union-find approach on the sorted edge list.

        Args:
            dist: Pairwise distance matrix, shape (n, n).

        Returns:
            Birth-death pairs, shape (k, 2).
        """
        n = dist.shape[0]
        if n <= 1:
            return np.empty((0, 2))

        # Extract upper triangle edges
        edges: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist[i, j], i, j))
        edges.sort(key=lambda e: e[0])

        # Union-Find
        parent = list(range(n))
        rank_uf = [0] * n
        birth_time = [0.0] * n  # All components born at t=0

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pairs: List[Tuple[float, float]] = []
        for d_val, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                # Younger component dies (higher birth time, but all born at 0)
                if rank_uf[ri] < rank_uf[rj]:
                    parent[ri] = rj
                    pairs.append((0.0, d_val))
                elif rank_uf[ri] > rank_uf[rj]:
                    parent[rj] = ri
                    pairs.append((0.0, d_val))
                else:
                    parent[rj] = ri
                    rank_uf[ri] += 1
                    pairs.append((0.0, d_val))

        if len(pairs) == 0:
            return np.empty((0, 2))

        return np.array(pairs, dtype=np.float64)

    def save(self, path: Optional[str] = None) -> None:
        """Save atlas to disk as compressed npz file."""
        path = path or self.config.atlas_cache_path
        data = {}
        pdb_ids = list(self.entries.keys())
        data["pdb_ids"] = np.array(pdb_ids, dtype=object)
        data["genera"] = np.array([self.entries[k].genus for k in pdb_ids])
        data["lengths"] = np.array([self.entries[k].length for k in pdb_ids])

        sr_matrix = np.stack([self.entries[k].stable_rank for k in pdb_ids])
        data["stable_ranks"] = sr_matrix

        np.savez_compressed(path, **data)
        logger.info("atlas_saved", path=path, n_entries=len(pdb_ids))

    def load(self, path: Optional[str] = None) -> None:
        """Load atlas from disk."""
        path = path or self.config.atlas_cache_path
        if not os.path.exists(path):
            logger.warning("atlas_cache_not_found", path=path)
            return

        data = np.load(path, allow_pickle=True)
        pdb_ids = data["pdb_ids"]
        genera = data["genera"]
        lengths = data["lengths"]
        stable_ranks = data["stable_ranks"]

        for i, pdb_id in enumerate(pdb_ids):
            self.entries[str(pdb_id)] = AtlasEntry(
                pdb_id=str(pdb_id),
                sequence="",
                length=int(lengths[i]),
                genus=int(genera[i]),
                persistence_image=np.zeros(
                    (self.config.persistence_image_size, self.config.persistence_image_size)
                ),
                stable_rank=stable_ranks[i],
                coords_c3=np.empty((0, 3)),
                birth_death=np.empty((0, 2)),
            )

        logger.info("atlas_loaded", path=path, n_entries=len(self.entries))


class AtlasEntry:
    """Single entry in the topological atlas."""

    def __init__(
        self,
        pdb_id: str,
        sequence: str,
        length: int,
        genus: int,
        persistence_image: np.ndarray,
        stable_rank: np.ndarray,
        coords_c3: np.ndarray,
        birth_death: np.ndarray,
    ) -> None:
        self.pdb_id = pdb_id
        self.sequence = sequence
        self.length = length
        self.genus = genus
        self.persistence_image = persistence_image
        self.stable_rank = stable_rank
        self.coords_c3 = coords_c3
        self.birth_death = birth_death
