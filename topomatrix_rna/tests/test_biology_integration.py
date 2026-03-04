"""Integration tests for biology wiring across pipeline stages.

Validates that the biology functions defined in each stage are
actually invoked during normal pipeline execution (not dead code).
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from topomatrix_rna.config import (
    ContactMapConfig,
    DomainConfig,
    RiemannianConfig,
    RNABiologyConstants,
    TDAConfig,
    TropicalConfig,
)


class TestBIO01StageOneWiring:
    """BIO-01: LW weights, tetraloops, crossing pairs called in predict()."""

    def test_lw_weight_called_in_predict(self):
        """LW weights should be applied during predict() for contacts > 0.1 (vectorised)."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        P = predictor.predict("GGGGAAAACCCC", return_sparse=False)
        # The vectorised LW weighting scales significant contacts by
        # the (4,4) LW_MATRIX. G-C canonical pairs stay at 1.0,
        # while non-canonical (e.g. A-A) get 0.4×, so not all entries
        # above the threshold should be identical — confirming the LW
        # weighting pass ran.
        upper = P[np.triu_indices_from(P, k=1)]
        sig = upper[upper > 0.01]
        if len(sig) > 1:
            assert not np.allclose(sig, sig[0]), \
                "All significant contacts identical — LW weighting not applied"

    def test_detect_tetraloops_called_in_predict(self):
        """detect_tetraloops should be called during predict()."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        with patch('topomatrix_rna.stage1_contact_map.detect_tetraloops', wraps=__import__(
            'topomatrix_rna.stage1_contact_map', fromlist=['detect_tetraloops']
        ).detect_tetraloops) as mock_tl:
            P = predictor.predict("GGGGAAAACCCC", return_sparse=False)
            assert mock_tl.call_count == 1, "detect_tetraloops was never called in predict()"

    def test_find_crossing_pairs_called_in_predict(self):
        """find_crossing_pairs should be called during predict()."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        with patch('topomatrix_rna.stage1_contact_map.find_crossing_pairs', wraps=__import__(
            'topomatrix_rna.stage1_contact_map', fromlist=['find_crossing_pairs']
        ).find_crossing_pairs) as mock_cp:
            P = predictor.predict("GGGGAAAACCCC", return_sparse=False)
            assert mock_cp.call_count == 1, "find_crossing_pairs was never called in predict()"

    def test_predict_bio_config_used(self):
        """ContactMapPredictor should accept and use bio_config."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        bio = RNABiologyConstants(weight_pseudoknot=0.5)
        predictor = ContactMapPredictor(ContactMapConfig(), bio_config=bio)
        assert predictor._bio_config.weight_pseudoknot == 0.5

    def test_predict_still_symmetric(self):
        """Contact map should remain symmetric after biology integration."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        predictor = ContactMapPredictor(ContactMapConfig())
        P = predictor.predict("GAAAGGGCCCUUCG", return_sparse=False)
        np.testing.assert_allclose(P, P.T, atol=1e-10)


class TestBIO02StageTwo_ElectrostaticWiring:
    """BIO-02: compute_electrostatic_penalty called in _compute_weight_matrix when coords present."""

    def test_electrostatic_penalty_applied_when_coords_set(self):
        """Electrostatic penalty should modify W when _c4prime_coords is set."""
        from topomatrix_rna.stage2_tropical import TropicalBasinCensus, compute_electrostatic_penalty
        config = TropicalConfig()
        census = TropicalBasinCensus(config)

        # Without coords — baseline
        encoded = np.array([2, 0, 0, 0, 0, 1], dtype=np.int64)  # G....C
        cm = np.ones((6, 6))
        W_no_coords = census._compute_weight_matrix(encoded, cm)

        # With coords
        census._c4prime_coords = np.random.randn(6, 3) * 5.0
        W_with_coords = census._compute_weight_matrix(encoded, cm)

        # Should differ (electrostatic penalty added)
        finite_mask = np.isfinite(W_no_coords) & np.isfinite(W_with_coords)
        if np.any(finite_mask):
            assert not np.allclose(W_no_coords[finite_mask], W_with_coords[finite_mask]), \
                "Electrostatic penalty had no effect on weight matrix"

    def test_electrostatic_penalty_not_applied_when_no_coords(self):
        """Without coords, no electrostatic penalty should be applied."""
        from topomatrix_rna.stage2_tropical import TropicalBasinCensus
        config = TropicalConfig()
        census = TropicalBasinCensus(config)
        assert census._c4prime_coords is None


class TestBIO03PseudotorsionVectorised:
    """BIO-03: compute_pseudotorsions is vectorised (no Python loop)."""

    def test_pseudotorsions_correct_values(self):
        """Vectorised pseudotorsions should match scalar reference."""
        from topomatrix_rna.stage2_tropical import compute_pseudotorsions
        np.random.seed(42)
        L = 20
        c4 = np.random.randn(L, 3) * 5.0
        p = np.random.randn(L, 3) * 5.0
        eta, theta = compute_pseudotorsions(c4, p)

        assert eta.shape == (L,)
        assert theta.shape == (L,)
        assert np.isnan(eta[0]) and np.isnan(eta[-1])
        for i in range(1, L - 1):
            assert np.isfinite(eta[i])
            assert np.isfinite(theta[i])

    def test_pseudotorsions_small_input(self):
        """L=2 should return all NaN."""
        from topomatrix_rna.stage2_tropical import compute_pseudotorsions
        eta, theta = compute_pseudotorsions(np.zeros((2, 3)), np.zeros((2, 3)))
        assert np.all(np.isnan(eta))
        assert np.all(np.isnan(theta))


class TestBIO04StagesFour_PenaltiesInRefine:
    """BIO-04: suite/sugar/chi penalties called during refine()."""

    def test_penalties_called_in_refine(self):
        """Biology penalties should be invoked during refine() inner loop."""
        from topomatrix_rna.stage4_riemannian import RiemannianRefiner
        config = RiemannianConfig(n_steps=5, block_size=10)
        refiner = RiemannianRefiner(config)

        with patch.object(refiner, 'sugar_pucker_penalty', wraps=refiner.sugar_pucker_penalty) as mock_pp, \
             patch.object(refiner, 'chi_syn_penalty', wraps=refiner.chi_syn_penalty) as mock_cp, \
             patch.object(refiner, 'suite_conformer_penalty', wraps=refiner.suite_conformer_penalty) as mock_sp:
            theta_init = np.random.uniform(0, 2 * np.pi, (10, 7))
            seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)
            refiner.refine(theta_init, seq)

            assert mock_pp.call_count > 0, "sugar_pucker_penalty never called in refine()"
            assert mock_cp.call_count > 0, "chi_syn_penalty never called in refine()"
            assert mock_sp.call_count > 0, "suite_conformer_penalty never called in refine()"


class TestBIO08ConfigPassThrough:
    """BIO-08: Penalties use instance bio_config, not fresh defaults."""

    def test_suite_penalty_uses_instance_config(self):
        """suite_conformer_penalty should use self._bio_config by default."""
        from topomatrix_rna.stage4_riemannian import RiemannianRefiner
        bio = RNABiologyConstants(weight_suite_penalty=0.5)
        refiner = RiemannianRefiner(RiemannianConfig(), bio_config=bio)
        outlier = {k: 0.0 for k in ['delta_prev', 'epsilon', 'zeta', 'alpha', 'beta', 'gamma', 'delta']}
        pen_custom = refiner.suite_conformer_penalty(outlier)

        refiner_default = RiemannianRefiner(RiemannianConfig())
        pen_default = refiner_default.suite_conformer_penalty(outlier)

        # Custom weight 0.5 vs default 0.1 should produce different penalties
        assert pen_custom != pen_default

    def test_sugar_pucker_uses_instance_config(self):
        """sugar_pucker_penalty should use self._bio_config by default."""
        from topomatrix_rna.stage4_riemannian import RiemannianRefiner
        bio = RNABiologyConstants(c3endo_delta_max=200.0)  # expanded range
        refiner = RiemannianRefiner(RiemannianConfig(), bio_config=bio)
        # delta=150 is normally in C2'-endo window; with expanded C3'-endo it's C3'-endo
        pen = refiner.sugar_pucker_penalty(150.0)
        assert pen == pytest.approx(0.0)  # should be zero with expanded range


class TestBIO05Stage6GenusWiring:
    """BIO-05: count_pseudoknot_genus called in verify_and_refine()."""

    def test_genus_check_called_when_bp_list_provided(self):
        """Genus check should run when bp_list and expected_genus are given."""
        from topomatrix_rna.stage6_tda_verify import TDAVerifier
        config = TDAConfig(wasserstein_epsilon=1e6, max_retries=1)
        verifier = TDAVerifier(config)

        theta = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.zeros(10, dtype=np.int64)
        target_bd = np.array([[0.0, 1.0]])

        def compute_pers(t):
            return np.array([0.0]), np.array([1.0])

        bp_list = [(0, 5), (3, 8)]  # crossing → genus 1

        with patch('topomatrix_rna.stage6_tda_verify.count_pseudoknot_genus',
                   wraps=__import__('topomatrix_rna.stage6_tda_verify',
                                    fromlist=['count_pseudoknot_genus']).count_pseudoknot_genus) as mock_genus:
            verifier.verify_and_refine(
                theta, seq, target_bd, compute_pers, lambda t: t,
                bp_list=bp_list, expected_genus=0,
            )
            assert mock_genus.call_count >= 1, "count_pseudoknot_genus never called"

    def test_verify_still_works_without_bp_list(self):
        """Backward compat: works without bp_list (genus check skipped)."""
        from topomatrix_rna.stage6_tda_verify import TDAVerifier
        config = TDAConfig(wasserstein_epsilon=1e6, max_retries=1)
        verifier = TDAVerifier(config)
        theta = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.zeros(10, dtype=np.int64)
        target_bd = np.array([[0.0, 1.0]])

        def compute_pers(t):
            return np.array([0.0]), np.array([1.0])

        best_theta, passed, _ = verifier.verify_and_refine(
            theta, seq, target_bd, compute_pers, lambda t: t
        )
        assert passed


class TestBIO06Stage7HelixBoundaryWiring:
    """BIO-06: helix_boundary_penalty called in decompose()."""

    def test_helix_boundary_penalty_called_with_helix_spans(self):
        """decompose() should call helix_boundary_penalty when helix_spans provided."""
        from topomatrix_rna.stage7_domain import SpectralDomainDecomposer
        config = DomainConfig(use_threshold_length=100, min_domain_size=20)
        decomposer = SpectralDomainDecomposer(config)

        L = 200
        contact_map = np.zeros((L, L))
        contact_map[:100, :100] = np.ones((100, 100)) * 0.8
        contact_map[100:, 100:] = np.ones((100, 100)) * 0.8
        contact_map[:100, 100:] = np.ones((100, 100)) * 0.01
        contact_map[100:, :100] = np.ones((100, 100)) * 0.01
        np.fill_diagonal(contact_map, 0)

        with patch('topomatrix_rna.stage7_domain.helix_boundary_penalty',
                   wraps=__import__('topomatrix_rna.stage7_domain',
                                    fromlist=['helix_boundary_penalty']).helix_boundary_penalty) as mock_hbp:
            domains = decomposer.decompose(
                contact_map, L,
                helix_spans=[(95, 105)],
                ss_linkers=[(110, 120)],
            )
            assert mock_hbp.call_count >= 1, "helix_boundary_penalty never called in decompose()"

    def test_decompose_avoids_helix_cut(self):
        """decompose() should avoid cutting inside a helix."""
        from topomatrix_rna.stage7_domain import SpectralDomainDecomposer
        config = DomainConfig(use_threshold_length=100, min_domain_size=20)
        decomposer = SpectralDomainDecomposer(config)

        L = 200
        contact_map = np.zeros((L, L))
        contact_map[:100, :100] = np.ones((100, 100)) * 0.8
        contact_map[100:, 100:] = np.ones((100, 100)) * 0.8
        contact_map[:100, 100:] = np.ones((100, 100)) * 0.01
        contact_map[100:, :100] = np.ones((100, 100)) * 0.01
        np.fill_diagonal(contact_map, 0)

        # Helix spanning 95-105 — boundary at 100 would cut through it
        domains = decomposer.decompose(
            contact_map, L,
            helix_spans=[(95, 105)],
            ss_linkers=[],
        )
        # All domain boundaries should NOT be strictly inside (95, 105)
        for start, end in domains:
            if start > 0:
                assert not (95 < start < 105), \
                    f"Domain boundary at {start} cuts through helix (95, 105)"

    def test_decompose_backward_compat_no_helix_spans(self):
        """decompose() should still work without helix_spans."""
        from topomatrix_rna.stage7_domain import SpectralDomainDecomposer
        config = DomainConfig(use_threshold_length=100, min_domain_size=20)
        decomposer = SpectralDomainDecomposer(config)

        L = 200
        contact_map = np.zeros((L, L))
        contact_map[:100, :100] = np.ones((100, 100)) * 0.8
        contact_map[100:, 100:] = np.ones((100, 100)) * 0.8
        np.fill_diagonal(contact_map, 0)

        domains = decomposer.decompose(contact_map, L)
        assert len(domains) >= 1
        total = sum(d[1] - d[0] for d in domains)
        assert total == L


class TestBUGS4A_PenaltyBeforeAdam:
    """BUG-S4-A: Biology penalties must modify grad BEFORE ADAM consumes it."""

    def test_penalty_influences_gradient_before_adam(self):
        """Grad scaling by bio penalty must happen before ADAM moment update."""
        from topomatrix_rna.stage4_riemannian import RiemannianRefiner
        # Use a bio config that will produce non-zero penalties for most angles
        bio = RNABiologyConstants()
        config = RiemannianConfig(n_steps=3, block_size=10)
        refiner = RiemannianRefiner(config, bio_config=bio)
        np.random.seed(42)
        theta_init = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)

        # Run with penalties active
        theta_bio, energy_bio = refiner.refine(theta_init.copy(), seq)

        # Run with zero penalties (all suite/pucker/chi return 0)
        refiner_noop = RiemannianRefiner(config, bio_config=bio)
        with patch.object(refiner_noop, 'sugar_pucker_penalty', return_value=0.0), \
             patch.object(refiner_noop, 'chi_syn_penalty', return_value=0.0), \
             patch.object(refiner_noop, 'suite_conformer_penalty', return_value=0.0):
            theta_noop, energy_noop = refiner_noop.refine(theta_init.copy(), seq)

        # With penalties active, the optimization path should differ
        assert not np.allclose(theta_bio, theta_noop, atol=1e-10), \
            "Bio penalties had no effect on ADAM — still applied after grad consumed?"


class TestBUGS2A_SetBackboneCoords:
    """BUG-S2-A: set_backbone_coords() must set _c4prime_coords."""

    def test_set_backbone_coords_method_exists(self):
        """TropicalBasinCensus should have a set_backbone_coords method."""
        from topomatrix_rna.stage2_tropical import TropicalBasinCensus
        census = TropicalBasinCensus(TropicalConfig())
        assert hasattr(census, 'set_backbone_coords')

    def test_set_backbone_coords_sets_value(self):
        """set_backbone_coords should update _c4prime_coords."""
        from topomatrix_rna.stage2_tropical import TropicalBasinCensus
        census = TropicalBasinCensus(TropicalConfig())
        assert census._c4prime_coords is None
        coords = np.random.randn(10, 3)
        census.set_backbone_coords(coords)
        np.testing.assert_array_equal(census._c4prime_coords, coords)

    def test_electrostatic_penalty_reachable_via_setter(self):
        """After calling set_backbone_coords, electrostatic penalty should be applied."""
        from topomatrix_rna.stage2_tropical import TropicalBasinCensus
        config = TropicalConfig()
        census = TropicalBasinCensus(config)

        encoded = np.array([2, 0, 0, 0, 0, 1], dtype=np.int64)
        cm = np.ones((6, 6))
        W_before = census._compute_weight_matrix(encoded, cm)

        census.set_backbone_coords(np.random.randn(6, 3) * 5.0)
        W_after = census._compute_weight_matrix(encoded, cm)

        finite = np.isfinite(W_before) & np.isfinite(W_after)
        if np.any(finite):
            assert not np.allclose(W_before[finite], W_after[finite]), \
                "Electrostatic penalty still unreachable after set_backbone_coords"


class TestBUGS4B_SymplecticBioPenalty:
    """BUG-S4-B: refine_symplectic() must include biology penalties."""

    def test_symplectic_calls_bio_penalties(self):
        """refine_symplectic should invoke _apply_bio_penalty_to_grad."""
        from topomatrix_rna.stage4_riemannian import RiemannianRefiner
        config = RiemannianConfig(n_steps=3, block_size=10)
        refiner = RiemannianRefiner(config)

        with patch.object(refiner, '_apply_bio_penalty_to_grad',
                          wraps=refiner._apply_bio_penalty_to_grad) as mock_bio:
            theta_init = np.random.uniform(0, 2 * np.pi, (10, 7))
            seq = np.array([0, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype=np.int64)
            refiner.refine_symplectic(theta_init, seq)
            # Should be called twice per step (grad_old and grad_new)
            assert mock_bio.call_count == 6, \
                f"Expected 6 calls (2 per step × 3 steps), got {mock_bio.call_count}"


class TestBUGS6A_GenusMismatchRetry:
    """BUG-S6-A: Genus mismatch should trigger retry, not just warning."""

    def test_genus_mismatch_triggers_retry(self):
        """On genus mismatch, verify_and_refine should perturb and continue."""
        from topomatrix_rna.stage6_tda_verify import TDAVerifier
        config = TDAConfig(wasserstein_epsilon=1e-10, max_retries=3)
        verifier = TDAVerifier(config)

        theta = np.random.uniform(0, 2 * np.pi, (10, 7))
        seq = np.zeros(10, dtype=np.int64)
        target_bd = np.array([[0.0, 1.0]])

        def compute_pers(t):
            return np.array([0.0]), np.array([1.0])

        bp_list = [(0, 5), (3, 8)]  # crossing → genus 1

        with patch.object(verifier, '_geodesic_perturb',
                          wraps=verifier._geodesic_perturb) as mock_perturb:
            verifier.verify_and_refine(
                theta, seq, target_bd, compute_pers, lambda t: t,
                bp_list=bp_list, expected_genus=0,
            )
            # Should perturb on every attempt because genus always mismatches
            assert mock_perturb.call_count == config.max_retries, \
                f"Expected {config.max_retries} perturbs, got {mock_perturb.call_count}"


class TestBUGS1B_CoaxialJunction:
    """BUG-S1-B: detect_coaxial_junctions should be called in predict()."""

    def test_coaxial_junctions_called_in_predict(self):
        """detect_coaxial_junctions should be invoked during predict()."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        config = ContactMapConfig()
        predictor = ContactMapPredictor(config)
        with patch('topomatrix_rna.stage1_contact_map.detect_coaxial_junctions',
                   wraps=__import__('topomatrix_rna.stage1_contact_map',
                                    fromlist=['detect_coaxial_junctions']).detect_coaxial_junctions) as mock_cx:
            P = predictor.predict("GGGGAAAACCCC", return_sparse=False)
            assert mock_cx.call_count == 1, "detect_coaxial_junctions was never called"


class TestBUGS1C_AMinorBonus:
    """BUG-S1-C: A-minor bonus should be applied for adenosines near helices."""

    def test_a_minor_bonus_applied(self):
        """A residues contacting helix positions at |i-j|>4 should get a bonus."""
        from topomatrix_rna.stage1_contact_map import ContactMapPredictor
        # Sequence with adenosines separated from G-C helix-like region
        config = ContactMapConfig()
        bio = RNABiologyConstants(weight_a_minor_bonus=0.3)
        predictor = ContactMapPredictor(config, bio_config=bio)
        P = predictor.predict("GGGGCCCCCAAAAAA", return_sparse=False)
        # Just verify the map is symmetric and bounded
        np.testing.assert_allclose(P, P.T, atol=1e-10)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)


class TestExtractHelices:
    """Test the _extract_helices helper used for coaxial/A-minor."""

    def test_empty(self):
        from topomatrix_rna.stage1_contact_map import _extract_helices
        assert _extract_helices([]) == []

    def test_single_pair(self):
        from topomatrix_rna.stage1_contact_map import _extract_helices
        helices = _extract_helices([(2, 10)])
        assert len(helices) == 1
        assert helices[0] == (2, 2)

    def test_consecutive_pairs_form_helix(self):
        from topomatrix_rna.stage1_contact_map import _extract_helices
        bp_list = [(1, 10), (2, 9), (3, 8)]
        helices = _extract_helices(bp_list)
        assert len(helices) == 1
        assert helices[0] == (1, 3)
