"""Data utilities for TOPOMATRIX-RNA pipeline.

CIF file parsing, CSV loaders, coordinate transforms, and sequence encoding.
Uses gemmi for CIF parsing when available, with a pure-Python fallback.

References:
    - gemmi: https://gemmi.readthedocs.io/
    - CIF format: https://www.iucr.org/resources/cif
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from .config import AtlasConfig, PipelineConfig

logger = structlog.get_logger(__name__)

# Nucleotide encoding: A=0, C=1, G=2, U=3
NUC_MAP: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


@dataclass
class RNAStructure:
    """Parsed RNA structure with C3' coordinates and metadata."""

    pdb_id: str
    sequence: str
    coords_c3: np.ndarray  # shape (L, 3) — C3' atom coordinates
    chain_id: str = "A"
    resolution: float = 0.0


@dataclass
class RNASequenceRecord:
    """A sequence record from CSV files."""

    seq_id: str
    sequence: str


@dataclass
class RNALabelRecord:
    """A label record (coordinates) from CSV files."""

    seq_id: str
    resname: str
    resseq: int
    chain_id: str
    x: float
    y: float
    z: float


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode RNA sequence string to integer array.

    Args:
        sequence: RNA sequence string (ACGU/ACGT).

    Returns:
        Integer array of shape (L,) with values 0-3.
    """
    encoded = np.zeros(len(sequence), dtype=np.int64)
    for i, c in enumerate(sequence.upper()):
        encoded[i] = NUC_MAP.get(c, 0)
    return encoded


def parse_cif_c3_coords(cif_path: str) -> Optional[RNAStructure]:
    """Parse C3' coordinates from a CIF file.

    Attempts to use gemmi for fast parsing. Falls back to a simple regex parser
    if gemmi is not available.

    Args:
        cif_path: Path to .cif file.

    Returns:
        RNAStructure or None if parsing fails.
    """
    pdb_id = os.path.splitext(os.path.basename(cif_path))[0]

    try:
        import gemmi

        st = gemmi.read_structure(cif_path)
        coords = []
        sequence_chars: List[str] = []
        chain_id = "A"

        for model in st:
            for chain in model:
                chain_id = chain.name
                for res in chain:
                    c3_atom = res.find_atom("C3'", "\x00")
                    if c3_atom:
                        pos = c3_atom.pos
                        coords.append([pos.x, pos.y, pos.z])
                        # Extract single-letter nucleotide code
                        resname = res.name.strip()
                        nuc = resname[-1] if resname else "A"
                        if nuc not in NUC_MAP:
                            nuc = "A"
                        sequence_chars.append(nuc)
                break  # First model only
            break

        if len(coords) == 0:
            logger.warning("no_c3_atoms", pdb_id=pdb_id, path=cif_path)
            return None

        return RNAStructure(
            pdb_id=pdb_id,
            sequence="".join(sequence_chars),
            coords_c3=np.array(coords, dtype=np.float64),
            chain_id=chain_id,
        )

    except ImportError:
        logger.debug("gemmi_not_available", fallback="regex_parser")
        return _parse_cif_fallback(cif_path, pdb_id)
    except Exception as e:
        logger.warning("cif_parse_error", pdb_id=pdb_id, error=str(e))
        return None


def _parse_cif_fallback(cif_path: str, pdb_id: str) -> Optional[RNAStructure]:
    """Fallback CIF parser using regex for _atom_site records.

    Looks for C3' atoms in ATOM records.
    """
    coords = []
    sequence_chars: List[str] = []
    chain_id = "A"

    try:
        with open(cif_path, "r") as f:
            in_atom_site = False
            column_names: List[str] = []
            for line in f:
                line = line.strip()
                if line.startswith("_atom_site."):
                    in_atom_site = True
                    col_name = line.split(".")[1].strip()
                    column_names.append(col_name)
                    continue
                if in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                    parts = line.split()
                    if len(parts) < len(column_names):
                        continue
                    col_map = {name: parts[i] for i, name in enumerate(column_names) if i < len(parts)}
                    atom_name = col_map.get("label_atom_id", "").strip("'\"")
                    if atom_name == "C3'":
                        try:
                            x = float(col_map.get("Cartn_x", "0"))
                            y = float(col_map.get("Cartn_y", "0"))
                            z = float(col_map.get("Cartn_z", "0"))
                            coords.append([x, y, z])
                            resname = col_map.get("label_comp_id", "A").strip()
                            nuc = resname[-1] if resname else "A"
                            if nuc not in NUC_MAP:
                                nuc = "A"
                            sequence_chars.append(nuc)
                            chain_id = col_map.get("label_asym_id", "A")
                        except (ValueError, KeyError):
                            continue
                elif in_atom_site and not line.startswith("_") and not line.startswith("#"):
                    if line == "" or line.startswith("loop_"):
                        in_atom_site = False
                        column_names = []

        if len(coords) == 0:
            return None

        return RNAStructure(
            pdb_id=pdb_id,
            sequence="".join(sequence_chars),
            coords_c3=np.array(coords, dtype=np.float64),
            chain_id=chain_id,
        )
    except Exception as e:
        logger.warning("cif_fallback_error", pdb_id=pdb_id, error=str(e))
        return None


def load_sequences_csv(csv_path: str) -> List[RNASequenceRecord]:
    """Load RNA sequences from CSV file.

    Expected format: id, sequence (with header row).

    Args:
        csv_path: Path to sequences CSV.

    Returns:
        List of RNASequenceRecord.
    """
    records: List[RNASequenceRecord] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("id", row.get("ID", "")).strip()
            sequence = row.get("sequence", row.get("Sequence", "")).strip()
            if seq_id and sequence:
                records.append(RNASequenceRecord(seq_id=seq_id, sequence=sequence))
    logger.info("loaded_sequences", path=csv_path, count=len(records))
    return records


def load_labels_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """Load ground-truth C3' coordinates from labels CSV.

    Expected format: id, resname, resseq, chain_id, x, y, z

    Args:
        csv_path: Path to labels CSV.

    Returns:
        Dict mapping sequence_id to coords array of shape (L, 3).
    """
    raw: Dict[str, List[Tuple[int, float, float, float]]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("id", row.get("ID", "")).strip()
            try:
                resseq = int(row.get("resseq", row.get("ResSeq", "0")))
                x = float(row.get("x", row.get("X", "0")))
                y = float(row.get("y", row.get("Y", "0")))
                z = float(row.get("z", row.get("Z", "0")))
            except (ValueError, KeyError):
                continue
            if seq_id not in raw:
                raw[seq_id] = []
            raw[seq_id].append((resseq, x, y, z))

    result: Dict[str, np.ndarray] = {}
    for seq_id, entries in raw.items():
        entries.sort(key=lambda e: e[0])
        coords = np.array([[e[1], e[2], e[3]] for e in entries], dtype=np.float64)
        result[seq_id] = coords

    logger.info("loaded_labels", path=csv_path, n_sequences=len(result))
    return result


def load_metadata_csv(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Load RNA metadata from CSV.

    Args:
        csv_path: Path to rna_metadata.csv.

    Returns:
        Dict mapping sequence_id to metadata dict.
    """
    metadata: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("sequence_id", row.get("pdb_id", "")).strip()
            if seq_id:
                metadata[seq_id] = dict(row)
    logger.info("loaded_metadata", path=csv_path, count=len(metadata))
    return metadata


def coords_to_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix from coordinates.

    Args:
        coords: Shape (N, 3).

    Returns:
        Distance matrix of shape (N, N).
    """
    N = coords.shape[0]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def kabsch_align(
    P: np.ndarray, Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kabsch algorithm for optimal rigid-body alignment.

    Finds rotation R and translation t that minimize RMSD between P and Q:
        P_aligned = (P - centroid_P) @ R + centroid_Q

    Reference: Kabsch W (1976) Acta Cryst A32:922-923

    Args:
        P: Predicted coordinates, shape (N, 3).
        Q: Target coordinates, shape (N, 3).

    Returns:
        Tuple of (aligned_P, rotation_matrix, rmsd).
    """
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    R = Vt.T @ sign_matrix @ U.T
    P_aligned = P_centered @ R + centroid_Q

    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

    return P_aligned, R, rmsd


def torsion_angle(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> float:
    """Compute dihedral/torsion angle between four 3D points.

    Args:
        p1, p2, p3, p4: 3D coordinate vectors of shape (3,).

    Returns:
        Torsion angle in radians [-π, π].
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)
