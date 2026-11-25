"""Permuted-label null for the closure fraction kappa.

kappa reads a capacity-driven change in the deficit D = R(random) - R(trained).
A probe fitted to permuted labels moves across the same ladder for reasons that
have nothing to do with the factor, so kappa needs its own control task, not just
selectivity at the top rung. This computes the missing role -- the random encoder
under the control task -- and reports

    kappa_perm(F) = 1 - D_perm(F, c4) / D_perm(F, c1),
    D_perm = R_perm(random encoder) - R_perm(trained encoder),

against which the reported kappa is read. The trained-side R_perm already ships in
stacks.npz; only the random-encoder side is new, and it is shared across every
Shapes3D cell because the random role is one array (verified: the `random` stacks
of color_strong and control_strong are bit-identical).

Uses the same code paths, splits, seeds and pinned probe config as run_sweep, so
the new role is commensurable with the shipped ones by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.registry import get_dataset
from ..data.splits import make_splits
from ..encoders.augmentations import eval_transform
from ..encoders.random_encoder import build_random_backbone_projector
from ..eval.extract import extract_features
from ..probes.instrument import stack_runs
from ..probes.run_sweep import _subsample
from ..utils.config import load_config
from ..utils.device import pick_device


def _build_datasets(spec, cfg, path: str, subsample: int, in_memory: bool):
    """Probe-split datasets, with run_sweep's probe-train subsample applied up front.

    run_sweep extracts the full 40,000-image probe-train split and then keeps
    ``rng(0).choice(len(H), n, replace=False)`` of the rows. Those indices do not
    depend on the features, so selecting the same dataset rows in the same order
    before extraction yields a bit-identical probe-train array at a fraction of the
    I/O. --verify checks that against the shipped random stack rather than assuming it.
    """
    splits = make_splits(spec.n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    names = ("probe_train", "probe_val", "probe_test")
    tr = splits["probe_train"]
    if subsample and len(tr) > subsample:
        idx = np.random.default_rng(0).choice(len(tr), subsample, replace=False)
        splits = {**splits, "probe_train": np.asarray(tr)[idx]}
    return {n: spec.cls(splits[n], transform=eval_transform(), path=path,
                        return_label=True, in_memory=in_memory) for n in names}, names


def random_stacks(dataset: str, seeds: list[int], *, subsample: int, steps: int,
                  epochs: int, batch_size: int, num_workers: int, device, config: str,
                  data_path: str | None, cache: Path | None,
                  in_memory: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(real, permuted) [S, F, R] stacks for the random encoder, one extraction each.

    Both roles are probed from the SAME extracted features on the SAME device, so
    the deficits D = R(random) - R(trained) and D_perm = R_perm(random) -
    R_perm(trained) carry the same device offset and it cancels in their comparison.
    """
    cfg = load_config(config)
    spec = get_dataset(dataset)
    path = data_path or spec.default_path
    datasets, names = _build_datasets(spec, cfg, path, subsample, in_memory)
    real_rows, perm_rows = [], []
    for rs in seeds:
        cr = cache / f"random_real_seed{rs}.npy" if cache else None
        cp = cache / f"random_perm_seed{rs}.npy" if cache else None
        if cr is not None and cr.exists() and cp.exists():
            real_rows.append(np.load(cr)); perm_rows.append(np.load(cp))
            print(f"[null]   random seed {rs}: cached", flush=True)
            continue
        bb, _ = build_random_backbone_projector(rs, device, out_dim=128)
        feats = {n: extract_features(bb, datasets[n], device, batch_size, num_workers)
                 for n in names}
        del bb
        kw = dict(device=device, epochs=epochs, steps=steps, factors=spec.factors)
        rr = stack_runs([(feats, rs)], permute=False, **kw)
        rp = stack_runs([(feats, rs)], permute=True, **kw)
        del feats
        real_rows.append(rr); perm_rows.append(rp)
        if cr is not None:
            cr.parent.mkdir(parents=True, exist_ok=True)
            np.save(cr, rr); np.save(cp, rp)
        print(f"[null]   random seed {rs}: probed (real + control task)", flush=True)
    return np.concatenate(real_rows), np.concatenate(perm_rows)


def kappa_rows(D: np.ndarray, factors: list[str], n_boot: int, rng) -> list[dict]:
    """kappa + paired seed-bootstrap CI from a per-seed deficit stack [S, F, R]."""
    n = D.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    out = []
    for i, f in enumerate(factors):
        dl, dt = D[:, i, 0], D[:, i, -1]
        defined = dl.mean() > 0
        num, den = dt[idx].mean(1), dl[idx].mean(1)
        kb = 1.0 - num / den
        kb = kb[np.isfinite(kb)]
        lo, hi = (np.percentile(kb, [2.5, 97.5]) if kb.size else (np.nan, np.nan))
        k = 1.0 - dt.mean() / dl.mean() if defined else float("nan")
        out.append(dict(factor=f, kappa_defined=bool(defined),
                        D_linear=float(dl.mean()), D_top=float(dt.mean()),
                        kappa=float(k),
                        kappa_ci95=[float(lo) if defined else float("nan"),
                                    float(hi) if defined else float("nan")]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="shapes3d")
    ap.add_argument("--cells", nargs="+", default=["color_strong", "control_strong"])
    ap.add_argument("--root", default="results/probes_pinned")
    ap.add_argument("--config", default="configs/probe/ladder.yaml")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--probe-steps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--out", default="results/hypotheses/kappa_null.json")
    ap.add_argument("--no-in-memory", dest="in_memory", action="store_false")
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the shipped random stack on one seed with the same "
                         "fast path, to prove the pre-applied subsample is equivalent")
    args = ap.parse_args()

    root = Path(args.root)
    meta = json.load(open(root / args.cells[0] / "meta.json"))
    factors = [f["name"] for f in meta["factors"]]
    seeds = meta["seeds"]["random"]
    for c in args.cells:
        m = json.load(open(root / c / "meta.json"))
        assert m["seeds"]["random"] == seeds, f"{c}: random seeds differ"
        assert (m["probe_train_size"], m["probe_steps"]) == (args.subsample, args.probe_steps), \
            f"{c}: pinned probe config differs from the requested one"
    a = np.load(root / args.cells[0] / "stacks.npz")["random"]
    for c in args.cells[1:]:
        b = np.load(root / c / "stacks.npz")["random"]
        assert np.array_equal(a, b), f"{c}: random stack differs; the role is not shared"

    device = pick_device(args.device)
    cache = Path("results/probes_pinned/_random_perm")
    kw = dict(subsample=args.subsample, steps=args.probe_steps, epochs=args.epochs,
              batch_size=args.batch_size, num_workers=args.num_workers, device=device,
              config=args.config, data_path=args.data_path, cache=cache,
              in_memory=args.in_memory)
    print(f"[null] {args.dataset} | device={device} | {len(seeds)} random encoders | "
          f"probe_train={args.subsample} steps={args.probe_steps}", flush=True)

    rr, rp = random_stacks(args.dataset, seeds, **kw)






    lin = float(np.abs(rr[:, :, 0] - a[:len(rr), :, 0]).max())
    full = float(np.abs(rr - a[:len(rr)]).max())
    print(f"\n[null] device check against the shipped random stack: "
          f"linear rung max |dev| = {lin:.2e}, all rungs = {full:.2e}")
    if lin > 5e-3:
        raise SystemExit("[null] FAIL: the recomputed random stack disagrees at the "
                         "LINEAR rung, so the splits or the subsample differ. Stop.")
    out_dev = {"linear_rung_max_dev": lin, "all_rungs_max_dev": full,
               "note": "shipped stacks are CUDA, these are this host's device; the "
                       "linear rung is the equivalence proof and both deficits below "
                       "use the SAME recomputed random role, so the offset cancels"}

    out = {"dataset": args.dataset, "n_boot": args.n_boot, "seeds": seeds,
           "probe_train_size": args.subsample, "probe_steps": args.probe_steps,
           "random_perm_mean": rp.mean(0).tolist(),
           "random_real_mean": rr.mean(0).tolist(), "factors": factors,
           "rungs": meta["rungs"], "device_check": out_dev, "cells": {}}
    for cell in args.cells:
        z = np.load(root / cell / "stacks.npz")
        D_real = rr - z["trained"]
        D_perm = rp - z["perm"]
        real = kappa_rows(D_real, factors, args.n_boot, np.random.default_rng(20260824))
        null = kappa_rows(D_perm, factors, args.n_boot, np.random.default_rng(20260824))
        rows = []
        for r, nl in zip(real, null):
            rows.append({**r, "kappa_perm": nl["kappa"], "kappa_perm_ci95": nl["kappa_ci95"],
                         "kappa_perm_defined": nl["kappa_defined"],
                         "D_perm_linear": nl["D_linear"], "D_perm_top": nl["D_top"]})
        out["cells"][cell] = rows
        print(f"\n== {cell} ==")
        for r in rows:
            k = (f"{r['kappa']:+.3f} [{r['kappa_ci95'][0]:+.3f},{r['kappa_ci95'][1]:+.3f}]"
                 if r["kappa_defined"] else "UNDEFINED".center(22))
            kp = (f"{r['kappa_perm']:+.3f} [{r['kappa_perm_ci95'][0]:+.3f},"
                  f"{r['kappa_perm_ci95'][1]:+.3f}]" if r["kappa_perm_defined"]
                  else "UNDEFINED".center(22))
            print(f"  {r['factor']:12s} kappa={k}   kappa_perm={kp}")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
