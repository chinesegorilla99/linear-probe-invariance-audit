"""Unit tests for the counterfactual displacement spectrum (newplan §5.2/§5.4).

Every fixture is a hand-built factorial grid or a synthetic field with a KNOWN
answer, so each of the six technical requirements is checked against ground truth
rather than against the implementation's own output. No dataset, no encoder, no
training.

Run:  python -m unittest tests.test_displacement -v
"""

import argparse
import unittest

import numpy as np

from src.probes.displacement import (
    run,
    analyze_role,
    bootstrap_stats,
    calibrate_m_star,
    certificate,
    context_scale,
    epsilon_m,
    grid_codes,
    grid_index,
    sample_contexts,
    spectrum_stats,
    shared_direction_fraction,
    sweep_rows,
    displacement_scale,
    whitener,
)

N_BOOT = 200


def toy_grid(radices=(3, 4, 2)):
    """Complete factorial grid, shuffled so nothing depends on row order."""
    mesh = np.meshgrid(*[np.arange(r) for r in radices], indexing="ij")
    labels = np.stack([m.ravel() for m in mesh], axis=1).astype(float)
    return labels[np.random.default_rng(0).permutation(labels.shape[0])]


class TestGrid(unittest.TestCase):
    def test_sweep_rows_hold_context_fixed(self):


        labels = toy_grid()
        codes, values = grid_codes(labels)
        lookup = grid_index(codes, values)
        col = 1
        ctx = sample_contexts(values, col, 16)
        rows = sweep_rows(ctx, lookup, values, col)
        self.assertEqual(rows.shape, (16, len(values[col])))
        for i in range(16):
            block = codes[rows[i]]
            for c in range(labels.shape[1]):
                if c == col:
                    np.testing.assert_array_equal(block[:, c], np.arange(len(values[c])))
                else:
                    self.assertEqual(len(set(block[:, c])), 1)
                    self.assertEqual(block[0, c], ctx[i, c])

    def test_incomplete_grid_is_refused(self):


        codes, values = grid_codes(toy_grid()[:-1])
        with self.assertRaises(ValueError):
            grid_index(codes, values)

    def test_constant_column_is_harmless(self):

        labels = np.concatenate([np.zeros((toy_grid().shape[0], 1)), toy_grid()], axis=1)
        codes, values = grid_codes(labels)
        self.assertEqual(len(values[0]), 1)
        grid_index(codes, values)

    def test_contexts_are_identical_across_calls(self):

        _, values = grid_codes(toy_grid())
        a = sample_contexts(values, 1, 32)
        np.testing.assert_array_equal(a, sample_contexts(values, 1, 32))
        self.assertFalse(np.array_equal(a, sample_contexts(values, 0, 32)))


class TestWhitening(unittest.TestCase):
    def test_whitening_is_exactly_invariant_under_an_orthogonal_reparameterization(self):


        rng = np.random.default_rng(1)
        d = 6
        Htr, Hs = rng.normal(size=(400, d)), rng.normal(size=(24, 5, d))
        Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        base = analyze_role({"f": Hs}, whitener(Htr), n_boot=N_BOOT)["f"]
        rot = analyze_role({"f": Hs @ Q.T}, whitener(Htr @ Q.T), n_boot=N_BOOT)["f"]
        for key in ("m", "rho", "r_eff", "m_total", "m_shared", "m_ctx", "d_prime"):
            self.assertAlmostEqual(base[key], rot[key], places=5, msg=key)

    def test_rank_capped_whitening_trades_exact_affine_invariance_for_conditioning(self):





        rng = np.random.default_rng(1)
        d = 64
        Htr, Hs = rng.normal(size=(400, d)), rng.normal(size=(24, 5, d))

        def deviation(var_fraction):
            devs = []
            for _ in range(5):
                A = rng.normal(size=(d, d))
                base = analyze_role({"f": Hs}, whitener(Htr, var_fraction), n_boot=N_BOOT)["f"]
                mapped = analyze_role({"f": Hs @ A.T}, whitener(Htr @ A.T, var_fraction),
                                      n_boot=N_BOOT)["f"]
                devs.append(abs(base["m_total"] - mapped["m_total"]) / base["m_total"])
            return float(np.mean(devs))

        capped, tighter, full = deviation(0.99), deviation(0.999), deviation(1.0)
        self.assertGreater(capped, 0.05)
        self.assertLess(tighter, capped)
        self.assertLess(full, 0.01)

    def test_underdetermined_whitening_is_refused(self):



        rng = np.random.default_rng(11)
        with self.assertRaises(ValueError):
            whitener(rng.normal(size=(20, 64)))

    def test_null_direction_is_projected_out_not_amplified(self):




        rng = np.random.default_rng(2)
        H = rng.normal(size=(200, 4))
        H[:, 3] = 0.0
        W, info = whitener(H, return_info=True)
        self.assertTrue(np.isfinite(W).all())
        self.assertLessEqual(info["rank"], 3)
        self.assertEqual(W.shape, (info["rank"], 4))
        self.assertLess(np.linalg.norm(W @ np.eye(4)[3]), 1e-8)


class TestSpectrum(unittest.TestCase):
    def test_single_direction_field_gives_rho_one(self):

        rng = np.random.default_rng(3)
        d = 8
        u = np.zeros(d); u[2] = 1.0
        amp = rng.uniform(0.5, 2.0, (40, 4, 1))
        Hs = np.cumsum(np.concatenate([np.zeros((40, 1, d)), amp * u], axis=1), axis=1)
        st = analyze_role({"f": Hs}, np.eye(d), n_boot=N_BOOT)["f"]
        self.assertGreater(st["rho"], 0.999)
        self.assertLess(st["r_eff"], 1.01)
        self.assertGreater(st["ci"]["rho"][0], 0.99)

    def test_isotropic_field_spreads_across_directions(self):

        rng = np.random.default_rng(4)
        d = 8
        Hs = np.cumsum(rng.normal(size=(300, 5, d)), axis=1)
        st = analyze_role({"f": Hs}, np.eye(d), n_boot=N_BOOT)["f"]
        self.assertLess(st["rho"], 0.30)
        self.assertGreater(st["r_eff"], d * 0.6)

    def test_held_out_rho_is_not_the_optimistic_in_sample_rho(self):


        rng = np.random.default_rng(5)
        d, n = 12, 30
        M = rng.normal(size=(n, d))
        st = spectrum_stats(M[: n // 2], M[n // 2:])
        self.assertGreater(st["rho_in_sample"], st["rho"])
        self.assertGreater(st["rho_in_sample"], 1.0 / d)

    def test_bootstrap_resamples_contexts_not_displacements(self):



        rng = np.random.default_rng(6)
        d = 5
        per_ctx = rng.normal(size=(20, 1, d))
        fit = rng.normal(size=(50, d))
        wide = bootstrap_stats(fit, np.repeat(per_ctx, 4, axis=1), n_boot=N_BOOT)
        single = bootstrap_stats(fit, np.repeat(per_ctx[:1], 4, axis=1), n_boot=N_BOOT)
        self.assertGreater(wide["m"][1] - wide["m"][0], 0.0)
        self.assertAlmostEqual(single["m"][1] - single["m"][0], 0.0, places=9)


class TestMonotonicity(unittest.TestCase):
    def test_monotone_and_folded_single_direction_fields_are_separated(self):



        d = 6
        u = np.zeros(d); u[0] = 1.0
        rng = np.random.default_rng(7)
        ramp = np.linspace(0, 1, 5)
        tent = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        mk = lambda prof: (prof[None, :, None] * u[None, None, :] * np.ones((30, 1, 1))
                           + rng.normal(0, 1e-4, (30, 5, d)))
        mono = analyze_role({"f": mk(ramp)}, np.eye(d), n_boot=N_BOOT)["f"]
        fold = analyze_role({"f": mk(tent)}, np.eye(d), n_boot=N_BOOT)["f"]

        self.assertGreater(mono["rho"], 0.99)
        self.assertGreater(fold["rho"], 0.99)
        self.assertGreater(mono["monotonicity"]["monotone_fraction"], 0.99)
        self.assertLess(fold["monotonicity"]["monotone_fraction"], 0.05)
        self.assertGreater(mono["monotonicity"]["mean_abs_spearman"], 0.99)
        self.assertLess(fold["monotonicity"]["mean_abs_spearman"], 0.6)


class TestDestructionScale(unittest.TestCase):
    """A12: the certificate needs an ABSOLUTE scale. A11 (b)'s deficit against the random
    encoder certified reduction, not destruction — these fixtures pin the difference.

    After whitening the embedding has unit variance in every direction, so a whitened
    displacement is measured in standard deviations of the representation itself. A probe
    picks its own direction, so that, not a variance share over all dimensions, is what
    bounds it.
    """

    @staticmethod
    def _field(scale, d=32, n_ctx=200, n_steps=4):
        u = np.zeros(d); u[0] = 1.0
        return np.ones((n_ctx, n_steps, 1)) * u[None, None, :] * scale

    def test_halved_but_decodable_factor_is_not_certified_destroyed(self):



        full = displacement_scale(self._field(1.0))
        half = displacement_scale(self._field(0.5))
        self.assertAlmostEqual(half["m_total"], 0.5, places=6)
        self.assertAlmostEqual(half["m_shared"], 0.5, places=6)
        self.assertGreater(half["m_total"], 0.1, "halved-but-decodable read as destroyed")
        self.assertLess(half["m_total"], full["m_total"])

    def test_collapsed_encoder_is_certified_destroyed(self):


        st = displacement_scale(self._field(0.0))
        self.assertLess(st["m_total"], 1e-9)
        self.assertLess(st["m_shared"], 1e-9)

    def test_context_dependent_movement_is_not_linearly_readable(self):



        d, n_ctx = 32, 200
        u = np.zeros(d); u[0] = 1.0
        sign = np.where(np.arange(n_ctx) % 2 == 0, 1.0, -1.0)[:, None, None]
        field = sign * u[None, None, :] * np.ones((n_ctx, 4, 1))
        st = displacement_scale(field)
        self.assertAlmostEqual(st["m_total"], 1.0, places=6)
        self.assertLess(st["m_shared"], 1e-9)
        self.assertLess(shared_direction_fraction(field), 0.05)

    def test_consistent_direction_field_is_fully_shared(self):
        field = self._field(1.0)
        self.assertAlmostEqual(displacement_scale(field)["m_shared"], 1.0, places=6)
        self.assertGreater(shared_direction_fraction(field), 0.99)


class TestCertificate(unittest.TestCase):
    """A15 (b): the eta / eta_star certificate that replaces 'm_total << 1'."""

    @staticmethod
    def _fixture(amplitude, d=64, n_ctx=128, n_vals=6, seed=4):
        """A12 (a)'s family h = z + a*v*u: known Bayes R2 = a^2/(a^2+1)."""
        rng = np.random.default_rng(seed)
        u = np.zeros(d); u[0] = 1.0
        vals = np.arange(n_vals, dtype=float)
        vals = (vals - vals.mean()) / vals.std()
        return rng.normal(size=(n_ctx, 1, d)) + amplitude * vals[None, :, None] * u

    def test_m_total_is_a_width_free_discriminability_index(self):





        m_by_d, ratio_by_d = [], []
        for d in (32, 128, 512):
            H = self._fixture(1.0, d=d, seed=5)
            sc = displacement_scale(np.diff(H, axis=1))
            m_by_d.append(sc["m_total"])
            ratio_by_d.append(sc["m_total"] / context_scale(H, np.eye(d)))
        self.assertLess(max(m_by_d) / min(m_by_d), 1.05)
        self.assertGreater(max(ratio_by_d) / min(ratio_by_d), 3.0)

    def test_a_decodable_ramp_is_not_certified_destroyed_and_a_collapsed_one_is(self):
        cal = calibrate_m_star(dims=64, n_ctx=128, seed=6)
        self.assertTrue(cal["separable"], "fixture classes overlap: no m_star")
        m_star = cal["m_star"]
        for amp, expected in ((10.0, "moves"), (3.0, "moves"),
                              (0.0, "eps_destroyed"), (1e-3, "eps_destroyed")):
            H = self._fixture(amp)
            sc = displacement_scale(np.diff(H, axis=1))
            v = certificate(sc["m_total"], sc["m_shared"],
                            context_scale(H, np.eye(H.shape[-1])), H.shape[-1], m_star)
            self.assertEqual(v["verdict"], expected, f"amplitude {amp}")

    def test_the_unadjudicated_band_is_reported_not_silently_split(self):



        cal = calibrate_m_star(dims=64, n_ctx=128, seed=6)
        lo, hi = cal["unadjudicated_band"]
        self.assertLess(lo, cal["m_star"])
        self.assertGreater(hi, cal["m_star"])
        self.assertGreater(hi / max(lo, 1e-12), 2.0)

    def test_no_verdict_without_a_calibration(self):


        v = certificate(0.0, 0.0, 10.0, rank=8, m_star=None)
        self.assertEqual(v["verdict"], "uncalibrated")
        self.assertIsNone(v["m_star"])

    def test_calibration_reports_overlap_instead_of_picking_a_number(self):


        cal = calibrate_m_star(dims=64, n_ctx=128, amplitudes=(0.5,), seed=7)
        self.assertIsNone(cal["m_star"])
        self.assertFalse(cal["separable"])

    def test_whitening_check_flags_a_subspace_that_is_not_unit_variance(self):



        rng = np.random.default_rng(9)
        k = 16
        H = rng.normal(size=(64, 5, k))
        good = certificate(1.0, 1.0, context_scale(H, np.eye(k)), rank=k)
        self.assertAlmostEqual(good["whitening_check"], 1.0, delta=0.15)
        bad = certificate(1.0, 1.0, context_scale(0.1 * H, np.eye(k)), rank=k)
        self.assertLess(bad["whitening_check"], 0.2)

    def test_context_reference_uses_a_derangement_and_holds_the_factor_fixed(self):


        rng = np.random.default_rng(8)
        H = rng.normal(size=(64, 5, 16))
        m_ctx = context_scale(H, np.eye(16))
        self.assertGreater(m_ctx, 1.0)
        self.assertAlmostEqual(m_ctx, np.sqrt(2 * 16), delta=1.5)


class TestEpsilonM(unittest.TestCase):
    def test_epsilon_m_is_the_null_quantile_and_needs_two_seeds(self):

        rng = np.random.default_rng(8)
        m = 1.0 + rng.normal(0, 0.05, 12)
        i, j = np.triu_indices(12, k=1)
        pool = np.concatenate([m[i] - m[j], m[j] - m[i]])
        self.assertAlmostEqual(epsilon_m(m), float(np.percentile(pool, 97.5)))
        self.assertTrue(np.isnan(epsilon_m([1.0])))

    def test_epsilon_m_does_not_shrink_with_control_sample_size(self):


        vals = []
        for s in (12, 24, 48):
            rng = np.random.default_rng(9)
            vals.append(epsilon_m(1.0 + rng.normal(0, 0.05, s)))
        for v in vals:
            self.assertGreater(v, 0.08)
            self.assertLess(v, 0.22)
        self.assertGreater(vals[-1], 0.8 * vals[0])

    def test_destruction_certificate_needs_a_deficit_beyond_the_null(self):



        rng = np.random.default_rng(10)
        null = 1.0 + rng.normal(0, 0.05, 12)
        eps, m_rand = epsilon_m(null), float(np.mean(null))
        self.assertGreater(m_rand - 0.0, eps)
        self.assertLess(m_rand - m_rand * 0.99, eps)


class TestBlindGuard(unittest.TestCase):
    """The confirmatory blind: trained-encoder targeted-factor values need A11 named."""

    def test_guard_refuses_a_trained_checkpoint_without_an_amendment(self):
        args = argparse.Namespace(trained_encoders=["results/encoders/x/backbone.pt"],
                                  amendment=None, config=None, dataset="shapes3d")
        with self.assertRaises(SystemExit) as cm:
            run(args)
        self.assertIn("BLIND GUARD", str(cm.exception))

    def test_guard_fires_before_any_other_work(self):



        args = argparse.Namespace(trained_encoders=["x/backbone.pt"], amendment=None)
        with self.assertRaises(SystemExit) as cm:
            run(args)
        self.assertIn("BLIND GUARD", str(cm.exception))

    def test_random_and_pixel_roles_need_no_flag(self):


        args = argparse.Namespace(trained_encoders=None, amendment=None)
        with self.assertRaises(Exception) as cm:
            run(args)
        self.assertNotIsInstance(cm.exception, SystemExit)


if __name__ == "__main__":
    unittest.main()
