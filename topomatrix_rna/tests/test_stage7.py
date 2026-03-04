"""Tests for Stage 7: Spectral Domain Decomposition."""

import numpy as np
import pytest

from topomatrix_rna.config import DomainConfig
from topomatrix_rna.stage7_domain import SE3DomainAssembler, SpectralDomainDecomposer


class TestSpectralDecomposition:
    """Tests for spectral domain decomposition."""

    def test_short_sequence_no_decomposition(self):
        """Sequences below threshold should not be decomposed."""
        config = DomainConfig(use_threshold_length=500)
        decomposer = SpectralDomainDecomposer(config)
        contact_map = np.random.rand(100, 100)
        domains = decomposer.decompose(contact_map, 100)
        assert len(domains) == 1
        assert domains[0] == (0, 100)

    def test_long_sequence_decomposition(self):
        """Long sequences should be decomposed into multiple domains."""
        config = DomainConfig(use_threshold_length=100, min_domain_size=20)
        decomposer = SpectralDomainDecomposer(config)

        # Create strongly block-diagonal contact map (2 clear domains)
        L = 200
        contact_map = np.zeros((L, L))
        contact_map[:100, :100] = np.ones((100, 100)) * 0.8
        contact_map[100:, 100:] = np.ones((100, 100)) * 0.8
        # Very weak inter-block connections
        contact_map[:100, 100:] = np.ones((100, 100)) * 0.01
        contact_map[100:, :100] = np.ones((100, 100)) * 0.01
        np.fill_diagonal(contact_map, 0)

        domains = decomposer.decompose(contact_map, L)
        assert len(domains) >= 2
        # All residues should be covered
        total = sum(d[1] - d[0] for d in domains)
        assert total == L

    def test_uniform_fallback(self):
        """Uniform decomposition should cover all residues."""
        config = DomainConfig(max_domain_size=50)
        decomposer = SpectralDomainDecomposer(config)
        domains = decomposer._uniform_decomposition(200)
        total = sum(d[1] - d[0] for d in domains)
        assert total == 200

    def test_block_diagonal_decomposition(self):
        """Block-diagonal contact map should yield at least 2 domains."""
        config = DomainConfig(use_threshold_length=50, min_domain_size=10)
        decomposer = SpectralDomainDecomposer(config)

        L = 100
        contact_map = np.zeros((L, L))
        # Two strongly connected blocks with zero inter-block contacts
        contact_map[:50, :50] = 1.0
        contact_map[50:, 50:] = 1.0
        np.fill_diagonal(contact_map, 0)

        domains = decomposer.decompose(contact_map, L)
        assert len(domains) >= 2
        total = sum(d[1] - d[0] for d in domains)
        assert total == L


class TestSE3Assembly:
    """Tests for SE(3) domain assembly."""

    def test_single_domain(self):
        """Single domain should return unchanged coordinates."""
        config = DomainConfig()
        assembler = SE3DomainAssembler(config)
        coords = np.random.randn(50, 3)
        result = assembler.assemble(
            [coords], [(0, 50)], np.zeros((50, 50))
        )
        np.testing.assert_allclose(result, coords)

    def test_two_domains(self):
        """Assembly of two domains should produce correct shape."""
        config = DomainConfig(se3_steps=5)
        assembler = SE3DomainAssembler(config)
        coords1 = np.random.randn(30, 3)
        coords2 = np.random.randn(20, 3)
        contact_map = np.random.rand(50, 50) * 0.1
        contact_map = (contact_map + contact_map.T) / 2

        result = assembler.assemble(
            [coords1, coords2], [(0, 30), (30, 50)], contact_map
        )
        assert result.shape == (50, 3)

    def test_assembly_no_nan(self):
        """Assembled coordinates must not contain any NaN values."""
        config = DomainConfig(se3_steps=2)
        assembler = SE3DomainAssembler(config)

        np.random.seed(5)
        coords1 = np.random.randn(20, 3) * 10
        coords2 = np.random.randn(15, 3) * 10
        L = 35
        contact_map = np.random.rand(L, L) * 0.2
        contact_map = (contact_map + contact_map.T) / 2

        result = assembler.assemble(
            [coords1, coords2], [(0, 20), (20, 35)], contact_map
        )
        assert not np.any(np.isnan(result)), "NaN found in assembled coordinates"
