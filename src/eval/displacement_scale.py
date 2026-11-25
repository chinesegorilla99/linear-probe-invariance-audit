"""Rank-stability companion to the displacement certificate.

The certificate ships m_total against an absolute threshold m* calibrated on a
rank-1 fixture. On that fixture m_total is invariant to the retained whitening
rank k; on the real displacement fields it is not, because the fields are
high-rank and mostly context-dependent, so the retained rank sets the scale.

This is a read-only companion. It does NOT re-scale the primary and does not
change any verdict (the stop rule forbids that after 2026-08-24). It reports,
beside every shipped value:

  * k and the attainable scale sqrt(2k) per role, which differ between the
    trained and random roles and so set different ceilings for the same number;
  * m_total / m_ctx, the within-role rank-normalized displacement;
  * m_total(trained) / m_total(random), the cross-role ratio at matched
    var_fraction, which is the statistic that survives the rank sweep;
  * the multiplicative margin from each verdict to the nearest band edge;
  * the fitted exponent of m_total in k across the preregistered var_fraction
    sweep, which is the number the calibration's width-free note assumes is 0.
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import numpy as np

DISP = Path("results/displacement")
SWEEP = {0.95: "color_strong_vf0.95", 0.99: "color_strong", 0.999: "color_strong_vf0.999"}


def _load(name: str) -> dict | None:
    p = DISP / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _per_factor(role: dict, field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for seed in role.values():
        for f, v in seed["factors"].items():
            out.setdefault(f, []).append(float(v[field]))
    return out


def cell_rows(doc: dict) -> dict:
    band_lo, band_hi = doc["unadjudicated_band"]
    rows: dict[str, dict] = {}
    ranks = {r: [v["whitening"]["rank"] for v in doc["roles"][r].values()]
             for r in doc["roles"]}
    for role in ("trained", "random"):
        if role not in doc["roles"]:
            continue
        R = doc["roles"][role]
        m = _per_factor(R, "m_total")
        mc = _per_factor(R, "m_ctx")
        ms = _per_factor(R, "m_shared")
        for f in m:
            rows.setdefault(f, {})[role] = {
                "m_total": st.mean(m[f]),
                "m_ctx": st.mean(mc[f]),
                "m_shared": st.mean(ms[f]),
                "m_over_ctx": st.mean(m[f]) / st.mean(mc[f]),
                "rank": st.mean(ranks[role]),
                "attainable_scale_sqrt_2k": (2 * st.mean(ranks[role])) ** 0.5,
            }
    for f, r in rows.items():
        if "trained" in r and "random" in r:
            r["m_trained_over_null"] = r["trained"]["m_total"] / r["random"]["m_total"]
        if "trained" in r:
            mt = r["trained"]["m_total"]
            r["margin_to_band_lower"] = mt / band_lo
            r["margin_to_band_upper"] = mt / band_hi
            verdicts = [v["factors"][f]["verdict"] for v in doc["roles"]["trained"].values()]
            r["verdicts"] = {v: verdicts.count(v) for v in sorted(set(verdicts))}
    return rows


def rank_scaling() -> dict:
    """Fit log m_total ~ alpha * log k over the preregistered var_fraction sweep."""
    out: dict[str, dict] = {}
    pts: dict[str, dict[str, list[float]]] = {}
    for vf, name in SWEEP.items():
        doc = _load(name)
        if doc is None:
            continue
        for role in ("trained", "random"):
            R = doc["roles"][role]
            k = st.mean(v["whitening"]["rank"] for v in R.values())
            for f, vals in _per_factor(R, "m_total").items():
                d = pts.setdefault(f"{role}/{f}", {"k": [], "m": [], "vf": []})
                d["k"].append(k); d["m"].append(st.mean(vals)); d["vf"].append(vf)
    for key, d in pts.items():
        if len(d["k"]) < 2 or min(d["m"]) <= 0:
            continue
        alpha, _ = np.polyfit(np.log(d["k"]), np.log(d["m"]), 1)
        out[key] = {"var_fractions": d["vf"], "rank": d["k"], "m_total": d["m"],
                    "exponent_alpha": float(alpha),
                    "note": "calibration assumes alpha = 0 (width-free)"}
    return out


def null_ratio_stability() -> dict:
    """m_total(trained)/m_total(random) across the sweep -- the rank-stable reading."""
    out: dict[str, dict[str, float]] = {}
    for vf, name in SWEEP.items():
        doc = _load(name)
        if doc is None:
            continue
        tr = _per_factor(doc["roles"]["trained"], "m_total")
        rn = _per_factor(doc["roles"]["random"], "m_total")
        for f in tr:
            out.setdefault(f, {})[str(vf)] = st.mean(tr[f]) / st.mean(rn[f])
    for f, d in out.items():
        v = list(d.values())
        d["spread_max_over_min"] = max(v) / min(v)
    return out


def _by_seed(role: dict, field: str) -> dict[str, dict[int, float]]:
    """Per-factor values keyed by seed index, so the two roles can be paired."""
    out: dict[str, dict[int, float]] = {}
    for key, rec in role.items():
        i = int(re.search(r"seed(\d+)$", key).group(1))
        for f, v in rec["factors"].items():
            out.setdefault(f, {})[i] = float(v[field])
    return out


def null_ratio_ci(draws: int = 4000, rng_seed: int = 0) -> dict:
    """Seed-bootstrap interval on m_total(trained)/m_total(random).

    Same resampling unit and estimator as the closure-fraction intervals: the
    seeds are resampled with the two roles paired and the statistic is the ratio
    of the resampled means. Read-only; no verdict depends on it.
    """
    rng = np.random.default_rng(rng_seed)
    out: dict[str, dict] = {}
    for vf, name in SWEEP.items():
        doc = _load(name)
        if doc is None:
            continue
        tr = _by_seed(doc["roles"]["trained"], "m_total")
        rn = _by_seed(doc["roles"]["random"], "m_total")
        for f in tr:
            seeds = sorted(set(tr[f]) & set(rn[f]))
            if not seeds:
                continue
            t = np.array([tr[f][s] for s in seeds])
            r = np.array([rn[f][s] for s in seeds])
            idx = rng.integers(0, len(seeds), size=(draws, len(seeds)))
            boot = t[idx].mean(axis=1) / r[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.setdefault(f, {})[str(vf)] = {
                "n_seeds": len(seeds), "draws": draws,
                "ratio": float(t.mean() / r.mean()),
                "ci95": [float(lo), float(hi)],
            }
    return out


def main() -> None:
    cells = ["color_strong", "control_strong", "position_strong",
             "color_strong_vf0.95", "color_strong_vf0.999",
             "control_strong_vf0.95", "control_strong_vf0.999",
             "position_strong_vf0.95", "position_strong_vf0.999"]
    out = {"kind": "displacement_scale_companion",
           "note": "read-only; does not re-scale the primary or change any verdict",
           "cells": {}, "rank_scaling": rank_scaling(),
           "null_ratio_stability": null_ratio_stability(),
           "null_ratio_ci": null_ratio_ci(),
           "sweep_coverage": {}}
    for c in cells:
        doc = _load(c)
        if doc is None:
            continue
        out["cells"][c] = {"var_fraction": doc["var_fraction"],
                           "unadjudicated_band": doc["unadjudicated_band"],
                           "m_star": doc["m_star"], "factors": cell_rows(doc)}
    for arm in ("color_strong", "control_strong", "position_strong"):
        out["sweep_coverage"][arm] = sorted(
            _load(n)["var_fraction"] for n in
            (arm, f"{arm}_vf0.95", f"{arm}_vf0.999") if _load(n))

    p = DISP / "scale_companion.json"
    p.write_text(json.dumps(out, indent=1))

    print("== rank scaling of m_total (calibration assumes exponent 0) ==")
    for k, v in sorted(out["rank_scaling"].items()):
        if "trained/" in k:
            print(f"  {k:26s} k={[round(x) for x in v['rank']]} "
                  f"m={[round(x, 4) for x in v['m_total']]}  alpha={v['exponent_alpha']:+.2f}")
    print("\n== trained/null ratio across the sweep (rank-stable reading) ==")
    for f, d in sorted(out["null_ratio_stability"].items()):
        vals = {k: round(v, 4) for k, v in d.items() if k != "spread_max_over_min"}
        print(f"  {f:12s} {vals}  spread={d['spread_max_over_min']:.2f}x")
    print("\n== trained/null ratio with a paired seed bootstrap (vf 0.99) ==")
    for f, d in sorted(out["null_ratio_ci"].items()):
        r = d.get("0.99")
        if r:
            lo, hi = r["ci95"]
            print(f"  {f:12s} ratio={r['ratio']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={r['n_seeds']}")

    print("\n== per-role scale mismatch and band margins (shipped vf) ==")
    for c in ("color_strong", "control_strong", "position_strong"):
        if c not in out["cells"]:
            continue
        print(f"  -- {c}")
        for f, r in out["cells"][c]["factors"].items():
            if "trained" not in r:
                continue
            t, n = r["trained"], r.get("random")
            line = (f"     {f:12s} k_tr={t['rank']:6.1f} k_rnd={n['rank']:6.1f}  "
                    f"m={t['m_total']:8.4f} m/m_ctx={t['m_over_ctx']:.5f}")
            if "m_trained_over_null" in r:
                line += f"  m/null={r['m_trained_over_null']:8.4f}"
            line += f"  band_lo_margin={r['margin_to_band_lower']:.3f}x  {r['verdicts']}"
            print(line)
    print(f"\n== var_fraction sweep coverage ==\n  {out['sweep_coverage']}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
