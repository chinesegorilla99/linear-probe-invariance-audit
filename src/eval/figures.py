"""Paper figures from the run_sweep contract (stacks.npz + meta.json per cell).

Consumes cells through the same loaders/statistics as the H1-H4 layer
(hypotheses.load_cell / analyze_cell / assemble + instrument's bootstrap), so
every plotted number matches the reported one. Works identically on real cells
(results/probes) and synthetic fixtures (results/_synthetic/<world>); any cell
whose meta.json carries "synthetic_world" stamps SYNTHETIC on every output.

Per root it renders (PNG + PDF + a CSV table twin per figure):

    g_ladder_<cell>   per-factor small multiples: G vs probe capacity, bootstrap
                      CI band, +-epsilon_G band, targeted factors highlighted
    h4_<cell>         targeted factors: recoverability of h vs projector vs the
                      random floor across the ladder (skipped when no targets)
    headline          the prereg 0 contrast: object hue (Color) vs x/y position
                      (Position), G across the ladder (needs both cells)
    verdicts          genuine-vs-artifact verdict matrix over (factor, cell)

    python -m src.eval.figures --root results/_synthetic/artifact \
        --out results/figures/_synthetic/artifact

Two further figures are rendered from the assembled artifacts rather than from the
cell stacks, so a plotted number cannot drift from the reported one. They are opt-in
per run and every expected cell, factor and role is asserted rather than skipped:

    kappa_dissociation    closure fraction per factor per arm, with seed-bootstrap
                          CI95, from results/hypotheses/kappa_ci.json
    displacement_certificate
                          m_total per factor per cell against m_star and the
                          unadjudicated band, from results/displacement/

    python -m src.eval.figures --root results/probes_pinned \
        --out results/figures/real \
        --kappa-json results/hypotheses/kappa_ci.json \
        --displacement-dir results/displacement
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..probes.hypotheses import (
    TARGETED_FACTORS,
    analyze_cell,
    assemble,
    discover_cells,
    load_cell,
)
from ..probes.instrument import EPS_FIXED, _boot_mean_ci, epsilon_g, paired_gain


BLUE, AQUA, YELLOW, ORANGE = "#2a78d6", "#1baf7a", "#eda100", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, CRITICAL = "#e1e0d9", "#c3c2b7", "#d03b3b"

VERDICT_STYLE = {
    "suppressed_across_ladder": (BLUE, "suppressed"),
    "invariant_across_ladder": (BLUE, "invariant"),
    "linear_invariance_artifact": (ORANGE, "artifact (flip)"),
    "recovered_at_all_capacities": (AQUA, "recovered"),
    "inconclusive_probe_driven": (GRID, "inconclusive"),
}
N_BOOT = 1000


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 7,
    })


LINE = dict(lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.5, solid_capstyle="round")


def _stamp(fig, synthetic: bool, note: str = "") -> None:
    """Footer row: methods note bottom-left, SYNTHETIC banner bottom-right."""
    if note:
        fig.text(0.01, 0.002, note, ha="left", va="bottom", color=MUTED, fontsize=7)
    if synthetic:
        fig.text(0.99, 0.002, "SYNTHETIC — PIPELINE VALIDATION ONLY",
                 ha="right", va="bottom", color=CRITICAL, fontsize=8, fontweight="bold")


def _text_on(fill: str) -> str:
    """Ink or white in-fill text, whichever clears more WCAG contrast."""
    def lin(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    L = sum(w * lin(int(fill[i:i + 2], 16)) for w, i in
            ((0.2126, 1), (0.7152, 3), (0.0722, 5)))
    return "#ffffff" if 1.05 / (L + 0.05) >= 3.0 else INK


def _save(fig, out: Path, name: str, rows: list[dict]) -> None:
    """PNG + PDF + the CSV table twin (the accessibility/verification channel)."""
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    if rows:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[figures] wrote {out / name}.{{png,pdf,csv}}")


def _cell_stats(cell, n_boot: int = N_BOOT) -> dict:
    """G with CIs, epsilon_G, and seed-mean h/projector/floor levels per cell."""
    g = paired_gain(cell.trained, cell.random_stats, n_boot=n_boot)
    eps = epsilon_g(cell.random_stats, n_boot=n_boot)
    eps = np.where(np.isnan(eps), EPS_FIXED, eps)
    levels = {name: _boot_mean_ci(arr, n_boot=n_boot)
              for name, arr in (("h", cell.trained), ("projector", cell.projector),
                                ("random floor", cell.random_stats))}
    return {"G": g, "eps": eps, "levels": levels}




def fig_g_ladder(cell, stats, out: Path, synthetic: bool) -> None:
    """Small multiples: one panel per factor, G vs rung with CI + epsilon band."""
    names = [f.name for f in cell.factors]
    targeted = set(TARGETED_FACTORS.get(cell.condition, ()))
    R = len(cell.rungs)
    x = np.arange(R)
    ncol = 3
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.15 * nrow + 0.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    G, eps, rows = stats["G"], stats["eps"], []
    r_tr, r_fl = stats["levels"]["h"], stats["levels"]["random floor"]
    for fi, (ax, fname) in enumerate(zip(axes, names)):
        hit = fname in targeted
        c = BLUE if hit else MUTED
        ax.fill_between(x, -eps[fi], eps[fi], color=GRID, alpha=0.65, lw=0)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.fill_between(x, G["lo"][fi], G["hi"][fi], color=c, alpha=0.10, lw=0)
        ax.plot(x, G["mean"][fi], color=c, **LINE)
        ax.set_title(fname + (" (targeted)" if hit else ""),
                     color=INK if hit else INK2,
                     fontweight="bold" if hit else "normal")
        ax.set_xticks(x, cell.rungs)
        ax.tick_params(axis="x", rotation=20)


        for ri, rname in enumerate(cell.rungs):
            rows.append({"factor": fname, "rung": rname, "targeted": hit,
                         "G_mean": round(float(G["mean"][fi, ri]), 4),
                         "G_lo": round(float(G["lo"][fi, ri]), 4),
                         "G_hi": round(float(G["hi"][fi, ri]), 4),
                         "epsilon_G": round(float(eps[fi, ri]), 4),
                         "R_trained_mean": round(float(r_tr[0][fi, ri]), 4),
                         "R_trained_lo": round(float(r_tr[1][fi, ri]), 4),
                         "R_trained_hi": round(float(r_tr[2][fi, ri]), 4),
                         "R_floor_mean": round(float(r_fl[0][fi, ri]), 4),
                         "R_floor_lo": round(float(r_fl[1][fi, ri]), 4),
                         "R_floor_hi": round(float(r_fl[2][fi, ri]), 4)})
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r * ncol].set_ylabel("encoder gain G")
    fig.suptitle(f"{cell.name} ({cell.dataset}) — G across the probe-capacity ladder, "
                 f"n={cell.trained.shape[0]} seeds", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, synthetic,
           "gray band: ±ε_G (invariance threshold) · color shading: 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save(fig, out, f"g_ladder_{cell.name}", rows)


def fig_h4(cell, stats, out: Path, synthetic: bool) -> None:
    """Targeted factors: recoverability of h vs projector vs the random floor."""
    targeted = [t for t in TARGETED_FACTORS.get(cell.condition, ())
                if t in [f.name for f in cell.factors]]
    if not targeted:
        return
    names = [f.name for f in cell.factors]
    x = np.arange(len(cell.rungs))
    series = (("h", BLUE), ("projector", AQUA), ("random floor", MUTED))
    fig, axes = plt.subplots(1, len(targeted),
                             figsize=(2.9 * len(targeted), 2.5), sharey=True)
    axes, rows = np.atleast_1d(axes), []
    for ax, t in zip(axes, targeted):
        fi = names.index(t)
        for sname, c in series:
            m, lo, hi = stats["levels"][sname]
            ax.fill_between(x, lo[fi], hi[fi], color=c, alpha=0.10, lw=0)
            ax.plot(x, m[fi], color=c, label=sname, **LINE)
            for ri, rname in enumerate(cell.rungs):
                rows.append({"factor": t, "series": sname, "rung": rname,
                             "mean": round(float(m[fi, ri]), 4),
                             "lo": round(float(lo[fi, ri]), 4),
                             "hi": round(float(hi[fi, ri]), 4)})
        ax.set_title(t, color=INK, fontweight="bold")
        ax.set_xticks(x, cell.rungs)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("recoverability")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{cell.name} — encoder h vs projector (H4), targeted factors",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, synthetic, "shading: 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _save(fig, out, f"h4_{cell.name}", rows)


def fig_headline(cells, all_stats, out: Path, synthetic: bool) -> None:
    """Prereg 0 contrast: object hue (Color) vs x/y position (Position)."""
    picks = []
    for cname, fname, c in (("color", "object_hue", BLUE),
                            ("position", "pos_x", AQUA), ("position", "pos_y", YELLOW)):
        cell = next((k for k in cells if k.condition == cname), None)
        if cell and fname in [f.name for f in cell.factors]:
            picks.append((f"{fname} ({cell.name})", cell, fname, c))
    if len(picks) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    rows = []
    for label, cell, fname, c in picks:
        fi = [f.name for f in cell.factors].index(fname)
        G = all_stats[cell.name]["G"]
        x = np.arange(len(cell.rungs))
        ax.fill_between(x, G["lo"][fi], G["hi"][fi], color=c, alpha=0.10, lw=0)

        ax.plot(x, G["mean"][fi], color=c, label=label, **LINE)
        for ri, rname in enumerate(cell.rungs):
            rows.append({"series": label, "rung": rname,
                         "G_mean": round(float(G["mean"][fi, ri]), 4),
                         "G_lo": round(float(G["lo"][fi, ri]), 4),
                         "G_hi": round(float(G["hi"][fi, ri]), 4)})
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(np.arange(len(picks[0][1].rungs)), picks[0][1].rungs)
    ax.set_xlim(-0.3, len(picks[0][1].rungs) - 0.4)
    ax.set_ylabel("encoder gain G")
    ax.legend(loc="upper left")
    ax.set_title("Headline contrast — targeted-factor G across the ladder",
                 loc="left", fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, synthetic, "shading: 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, "headline", rows)


def fig_verdicts(study, out: Path, synthetic: bool) -> None:
    """Verdict matrix over (factor row, cell column), flip carried in the label."""
    table = study["verdict_table"]
    cols = sorted({t["cell"] for t in table})
    rows_f = sorted({t["factor"] for t in table})
    by = {(t["factor"], t["cell"]): t for t in table}
    fig, ax = plt.subplots(figsize=(1.0 + 1.75 * len(cols), 0.8 + 0.42 * len(rows_f)))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows_f))
    ax.invert_yaxis()
    ax.axis("off")
    csv_rows = []
    for yi, fname in enumerate(rows_f):
        ax.text(-0.08, yi + 0.5, fname, ha="right", va="center", color=INK2, fontsize=7.5)
        for xi, cname in enumerate(cols):
            t = by.get((fname, cname))
            if t is None:
                continue
            fill, label = VERDICT_STYLE.get(t["verdict"], (GRID, t["verdict"]))

            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=fill,
                                       edgecolor=SURFACE, lw=2))
            ax.text(xi + 0.5, yi + 0.5, label, ha="center", va="center",
                    fontsize=7, color=_text_on(fill))
            csv_rows.append({"factor": fname, "cell": cname, "verdict": t["verdict"],
                             "case_linear": t["case_linear"], "case_top": t["case_top"],
                             "flip_two_sided": t["flip_two_sided"],

                             "verdict_frozen": t["verdict_frozen"],
                             "flip_primary": t["flip_primary"]})
    for xi, cname in enumerate(cols):
        ax.text(xi + 0.5, -0.15, cname, ha="center", va="bottom",
                color=INK, fontsize=8, fontweight="bold")
    n_flips = study["headline_flip_count"]["two_sided"]["n_flips"]
    ax.set_title(f"Verdicts per (factor, cell) — {n_flips} two-sided flip(s)"
                 + (" · provisional grid" if study["provisional"] else ""),
                 loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=24)
    _stamp(fig, synthetic)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out, "verdicts", csv_rows)






ARM_STYLE = {
    "control_strong":  (BLUE,   "o", "control-aug (minimal)"),
    "color_strong":    (ORANGE, "s", "color, strong"),
    "color_weak":      (AQUA,   "^", "color, weak (exploratory)"),
    "position_strong": (YELLOW, "D", "position, strong"),
}
SHAPES3D_ARMS = ("control_strong", "color_strong", "color_weak")
DSPRITES_ARMS = ("position_strong",)
CONFIRMATORY_CELLS = ("color_strong", "control_strong", "position_strong")
TARGETED_READOUTS = {
    "color_strong": ("floor_hue", "wall_hue", "object_hue"),
    "color_weak": ("floor_hue", "wall_hue", "object_hue"),
    "position_strong": ("pos_x", "pos_y"),
    "control_strong": (),
}





KAPPA_UNDEF_X = -0.50
KAPPA_BREAK_X = -0.40


KAPPA_XLIM = (KAPPA_UNDEF_X - 0.10, 1.02)
DISP_XLIM = (0.03, 60.0)


def _factor_rows(rows: list[dict], cell: str) -> dict:
    """factor -> row, asserting the artifact carries a row for every factor once."""
    out: dict[str, dict] = {}
    for r in rows:
        f = r["factor"]
        if f in out:
            raise ValueError(f"kappa artifact: duplicate factor {cell}:{f}")
        out[f] = r
    if not out:
        raise ValueError(f"kappa artifact: cell {cell} carries no factor rows")
    return out


def fig_kappa(kappa_json: Path, out: Path) -> None:
    """Closure fraction per factor per arm: the Claim-2 dissociation.

    Reads results/hypotheses/kappa_ci.json (written by src.eval.kappa_ci) rather
    than recomputing, so the plotted interval is the reported interval. Every
    confirmatory cell must be present and every factor of every plotted arm must
    resolve; a missing one raises instead of leaving a gap in the panel.
    """
    kappa_json = Path(kappa_json)
    if not kappa_json.exists():
        raise FileNotFoundError(f"kappa artifact not found: {kappa_json}")
    data = json.loads(kappa_json.read_text())
    missing = [c for c in CONFIRMATORY_CELLS if c not in data]
    if missing:
        raise ValueError(f"kappa artifact {kappa_json} is missing cells: {missing}")
    by_cell = {c: _factor_rows(rows, c) for c, rows in data.items()}

    for arm in (*SHAPES3D_ARMS, *DSPRITES_ARMS):
        if arm not in by_cell:
            print(f"[figures] kappa_dissociation: optional arm {arm!r} absent from "
                  f"{kappa_json}; it is omitted from the panel")
    panels = [
        ("Shapes3D", [a for a in SHAPES3D_ARMS if a in by_cell], "color_strong"),
        ("dSprites", [a for a in DSPRITES_ARMS if a in by_cell], "position_strong"),
    ]
    resolved = []
    for title, arms, anchor in panels:
        if not arms:
            raise ValueError(f"kappa figure: no arms available for the {title} panel")



        union = dict.fromkeys(by_cell[anchor])
        for arm in arms:
            union.update(dict.fromkeys(by_cell[arm]))
        factors = list(union)
        for arm in arms:
            absent = [f for f in factors if f not in by_cell[arm]]
            if absent:
                raise ValueError(
                    f"kappa figure: {arm} has no row for {absent} but shares the "
                    f"{title} panel; refusing to plot a partial panel")
        resolved.append((title, arms, factors))
    panels = resolved

    heights = [len(f) for _, _, f in panels]


    fig, all_axes = plt.subplots(
        3, 1, figsize=(6.0, 0.235 * sum(heights) + 1.05),
        gridspec_kw={"height_ratios": [*heights, 1.05]})
    axes, legend_ax = all_axes[:2], all_axes[2]
    axes[0].sharex(axes[1])
    axes[0].tick_params(labelbottom=False)
    legend_ax.axis("off")
    rows_csv: list[dict] = []

    for ax, (title, arms, factors) in zip(axes, panels):
        n = len(factors)

        offsets = np.linspace(0.22, -0.22, len(arms)) if len(arms) > 1 else [0.0]
        ax.axvspan(KAPPA_UNDEF_X - 0.06, KAPPA_UNDEF_X + 0.06,
                   color=GRID, alpha=0.9, lw=0, zorder=0)
        ax.axvline(KAPPA_BREAK_X, color=BASELINE, lw=1, zorder=1)
        ax.axvline(0.0, color=INK2, lw=1.2, zorder=2)
        for arm, dy in zip(arms, offsets):
            colour, marker, label = ARM_STYLE[arm]
            targeted = TARGETED_READOUTS[arm]
            for fi, fname in enumerate(factors):
                r = by_cell[arm][fname]
                y = (n - 1 - fi) + dy
                defined = bool(r["kappa_defined"])
                if defined:
                    k = float(r["kappa"])
                    lo, hi = (float(v) for v in r["kappa_ci95"])



                    for label_, v in (("kappa", k), ("ci_lo", lo), ("ci_hi", hi)):
                        if not math.isfinite(v):
                            raise ValueError(
                                f"kappa figure: {arm}:{fname} {label_} is {v}, but "
                                f"the row is marked kappa_defined")
                        if not (KAPPA_XLIM[0] <= v <= KAPPA_XLIM[1]):
                            raise ValueError(
                                f"kappa figure: {arm}:{fname} {label_}={v:.4f} falls "
                                f"outside the axis window {KAPPA_XLIM}; widen the "
                                f"window rather than letting the point be clipped")
                    ax.plot([lo, hi], [y, y], color=colour, lw=2,
                            solid_capstyle="round", zorder=3)
                    ax.plot([k], [y], color=colour, marker=marker, ms=6,
                            mec=SURFACE, mew=1.2, ls="none", zorder=4)
                else:
                    k = lo = hi = float("nan")
                    ax.plot([KAPPA_UNDEF_X], [y], color=colour, marker=marker,
                            ms=6, mfc="none", mew=1.4, ls="none", zorder=4)
                rows_csv.append({
                    "panel": title, "cell": arm, "factor": fname,
                    "targeted": fname in targeted,
                    "exploratory": bool(r["exploratory"]),
                    "kappa_defined": defined,
                    "kappa": "" if not defined else round(k, 4),
                    "kappa_ci95_lo": "" if not defined else round(lo, 4),
                    "kappa_ci95_hi": "" if not defined else round(hi, 4),
                    "D_linear": round(float(r["D_linear"]), 4),
                    "D_top": round(float(r["D_top"]), 4),
                    "S_top": round(float(r["S_top"]), 4)})
        ax.set_yticks(range(n))
        ax.set_yticklabels(list(reversed(factors)))
        any_targeted = set().union(*(set(TARGETED_READOUTS[a]) for a in arms))
        for lab in ax.get_yticklabels():
            if lab.get_text() in any_targeted:
                lab.set_fontweight("bold")
                lab.set_color(INK)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(*KAPPA_XLIM)
        ax.grid(axis="x")
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=INK)

    axes[-1].set_xlabel(r"closure fraction  $\kappa = 1 - D(\mathrm{top})\,/\,D(\mathrm{linear})$")
    ticks = [KAPPA_UNDEF_X, -0.25, 0.0, 0.25, 0.50, 0.75, 1.00]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(["undef.", "-0.25", "0", "0.25", "0.50", "0.75", "1.00"])
    handles = [plt.Line2D([], [], color=ARM_STYLE[a][0], marker=ARM_STYLE[a][1],
                          ms=6, lw=2, mec=SURFACE, mew=1.2, label=ARM_STYLE[a][2])
               for a in ("control_strong", "color_strong", "color_weak", "position_strong")
               if a in by_cell]
    legend_ax.legend(handles=handles, loc="upper center", ncol=2,
                     handletextpad=0.6, columnspacing=1.8, borderpad=0)
    fig.suptitle("Closure fraction by factor and augmentation condition, 12 seeds",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, False,
           "bars: 95% seed-bootstrap CI, 4000 paired draws · open marker in the "
           "left strip: kappa undefined because D(linear) <= 0\n"
           f"source: {kappa_json}")
    foot = 0.30 / fig.get_figheight()
    fig.tight_layout(rect=(0, foot, 1, 0.955))
    _save(fig, out, "kappa_dissociation", rows_csv)


def _displacement_cell(path: Path, role: str) -> tuple[dict, dict]:
    """(per-factor per-seed m_total, per-factor verdict counts) for one role."""
    if not path.exists():
        raise FileNotFoundError(f"displacement artifact not found: {path}")
    d = json.loads(path.read_text())
    if role not in d["roles"]:
        raise ValueError(f"{path}: role {role!r} absent; have {sorted(d['roles'])}")
    seeds = d["roles"][role]
    if not seeds:
        raise ValueError(f"{path}: role {role!r} carries no seeds")
    per_factor: dict[str, list[float]] = {}
    verdicts: dict[str, dict[str, int]] = {}
    for sk, rec in seeds.items():
        for fname, r in rec["factors"].items():
            m = float(r["m_total"])
            if not math.isfinite(m):
                raise ValueError(f"{path}: role {role!r} seed {sk} factor {fname} "
                                 f"has non-finite m_total {m}")
            per_factor.setdefault(fname, []).append(m)
            verdicts.setdefault(fname, {})
            verdicts[fname][r["verdict"]] = verdicts[fname].get(r["verdict"], 0) + 1
    n = len(seeds)
    ragged = {f: len(v) for f, v in per_factor.items() if len(v) != n}
    if ragged:
        raise ValueError(f"{path}: role {role!r} has {n} seeds but ragged factors {ragged}")
    return per_factor, verdicts


def fig_displacement(disp_dir: Path, out: Path) -> None:
    """m_total against m_star and the unadjudicated band: the Claim-3 certificate.

    Reads results/displacement/*.json. Every cell file, the two random-encoder
    reference files and the var_fraction sensitivity file must exist; the
    reference null is cross-checked against each cell's own random role and a
    disagreement raises rather than being plotted.
    """
    disp_dir = Path(disp_dir)
    cells = {"color_strong": "shapes3d", "control_strong": "shapes3d",
             "position_strong": "dsprites"}
    loaded, nulls, m_star, band, m_star_from = {}, {}, None, None, None
    for cell, dataset in cells.items():
        path = disp_dir / f"{cell}.json"
        meta = json.loads(path.read_text()) if path.exists() else None
        if meta is None:
            raise FileNotFoundError(f"displacement artifact not found: {path}")
        if meta.get("var_fraction") != 0.99:
            raise ValueError(f"{path}: expected the shipped var_fraction 0.99, "
                             f"got {meta.get('var_fraction')}")
        loaded[cell] = _displacement_cell(path, "trained")
        this_star = float(meta["m_star"])
        this_band = [float(v) for v in meta["unadjudicated_band"]]


        if m_star is None:
            m_star, band, m_star_from = this_star, this_band, path
        elif (this_star, this_band) != (m_star, band):
            raise ValueError(
                f"{path} carries m_star={this_star} band={this_band} but "
                f"{m_star_from} carries m_star={m_star} band={band}; the cells are "
                f"calibrated against different thresholds")
    for dataset in ("shapes3d", "dsprites"):
        nulls[dataset] = _displacement_cell(disp_dir / f"reference_{dataset}.json", "random")


    for cell, dataset in cells.items():
        own, _ = _displacement_cell(disp_dir / f"{cell}.json", "random")
        ref, _ = nulls[dataset]
        if set(own) != set(ref):
            raise ValueError(f"{cell}: random role factors {sorted(own)} differ from "
                             f"reference_{dataset} {sorted(ref)}")
        for fname, vals in ref.items():
            mine = own[fname]
            if len(mine) != len(vals):
                raise ValueError(f"{cell}: random role has {len(mine)} seeds for "
                                 f"{fname}, reference_{dataset} has {len(vals)}")


            for i, (a, b) in enumerate(zip(vals, mine)):
                if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(
                        f"{cell}: random role disagrees with reference_{dataset} on "
                        f"{fname} at seed slot {i} ({b} vs {a}); refusing to plot a "
                        f"mixed null")
    sens_path = disp_dir / "color_strong_vf0.999.json"
    sens, _ = _displacement_cell(sens_path, "trained")

    panels = []
    for title, dataset, arms, anchor in (
            ("Shapes3D", "shapes3d", ("control_strong", "color_strong"), "color_strong"),
            ("dSprites", "dsprites", ("position_strong",), "position_strong")):
        union = dict.fromkeys(loaded[anchor][0])
        for arm in arms:
            union.update(dict.fromkeys(loaded[arm][0]))
        union.update(dict.fromkeys(nulls[dataset][0]))
        factors = list(union)
        for arm in arms:
            absent = [f for f in factors if f not in loaded[arm][0]]
            if absent:
                raise ValueError(f"displacement figure: {arm} has no {absent} but "
                                 f"shares the {title} panel")
        absent = [f for f in factors if f not in nulls[dataset][0]]
        if absent:
            raise ValueError(f"displacement figure: reference_{dataset} has no {absent}")
        panels.append((title, dataset, arms, factors))

    heights = [len(f) for _, _, _, f in panels]
    fig, all_axes = plt.subplots(
        3, 1, figsize=(6.0, 0.255 * sum(heights) + 1.20),
        gridspec_kw={"height_ratios": [*heights, 1.35]})
    axes, legend_ax = all_axes[:2], all_axes[2]
    axes[0].sharex(axes[1])
    axes[0].tick_params(labelbottom=False)
    legend_ax.axis("off")
    rows_csv: list[dict] = []
    rng = np.random.default_rng(20260906)

    def draw(ax, y, vals, colour, marker, filled=True, who=""):


        for v in vals:
            if not (DISP_XLIM[0] <= v <= DISP_XLIM[1]):
                raise ValueError(f"displacement figure: {who} m_total={v:.4f} falls "
                                 f"outside the axis window {DISP_XLIM}; widen the "
                                 f"window rather than letting the point be clipped")
        jit = rng.uniform(-0.055, 0.055, size=len(vals))
        ax.plot(vals, y + jit, color=colour, marker=".", ms=2.6, ls="none",
                alpha=0.55, zorder=3)
        mean = float(np.mean(vals))
        ax.plot([mean], [y], color=colour, marker=marker, ms=6.5, ls="none",
                mec=SURFACE if filled else colour,
                mfc=colour if filled else "none", mew=1.3, zorder=4)
        return mean

    for ax, (title, dataset, arms, factors) in zip(axes, panels):
        n = len(factors)
        ax.axvspan(band[0], band[1], color=GRID, alpha=0.85, lw=0, zorder=0)

        ax.axvline(m_star, ymin=0.08, ymax=1.0, color=INK2, lw=1.2, zorder=2)
        slots = ["__null__", *arms]
        offsets = np.linspace(0.24, -0.24, len(slots))
        for slot, dy in zip(slots, offsets):
            for fi, fname in enumerate(factors):
                y = (n - 1 - fi) + dy
                if slot == "__null__":
                    vals = nulls[dataset][0][fname]
                    verd = nulls[dataset][1][fname]
                    mean = draw(ax, y, vals, MUTED, "o", filled=False,
                                who=f"reference_{dataset}:{fname}")
                    label, colour_name, expl = "random encoder (null)", MUTED, False
                else:
                    vals = loaded[slot][0][fname]
                    verd = loaded[slot][1][fname]
                    colour, marker, _ = ARM_STYLE[slot]
                    mean = draw(ax, y, vals, colour, marker, who=f"{slot}:{fname}")
                    label, expl = slot, False
                rows_csv.append({
                    "panel": title, "series": label, "factor": fname,
                    "var_fraction": 0.99, "n_seeds": len(vals),
                    "m_total_mean": round(mean, 4),
                    "m_total_min": round(min(vals), 4),
                    "m_total_max": round(max(vals), 4),
                    "verdicts": "; ".join(f"{k}={v}" for k, v in sorted(verd.items()))})


        if dataset == "shapes3d":
            for fi, fname in enumerate(factors):
                if fname not in ("floor_hue", "wall_hue", "object_hue"):
                    continue
                vals = sens[fname]
                y = (n - 1 - fi) + offsets[list(slots).index("color_strong")] - 0.17
                for v in vals:
                    if not (DISP_XLIM[0] <= v <= DISP_XLIM[1]):
                        raise ValueError(
                            f"displacement figure: {sens_path.name}:{fname} "
                            f"m_total={v:.4f} falls outside {DISP_XLIM}")
                mean = float(np.mean(vals))
                ax.plot([mean], [y], color=ORANGE, marker="s", ms=5, ls="none",
                        mfc="none", mew=1.3, zorder=4)
                rows_csv.append({
                    "panel": title, "series": "color_strong (sensitivity)",
                    "factor": fname, "var_fraction": 0.999, "n_seeds": len(vals),
                    "m_total_mean": round(mean, 4),
                    "m_total_min": round(min(vals), 4),
                    "m_total_max": round(max(vals), 4),
                    "verdicts": "unadjudicated=%d" % len(vals)})
        ax.set_xscale("log")
        ax.set_yticks(range(n))
        ax.set_yticklabels(list(reversed(factors)))
        targeted = set().union(*(set(TARGETED_READOUTS[a]) for a in arms))
        for lab in ax.get_yticklabels():
            if lab.get_text() in targeted:
                lab.set_fontweight("bold")
                lab.set_color(INK)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(*DISP_XLIM)
        ax.grid(axis="x")
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=INK)

    axes[0].annotate(r"$m^{*}$", xy=(m_star, len(panels[0][3]) - 0.55),
                     xytext=(3, 1), textcoords="offset points",
                     color=INK2, fontsize=7.5, ha="left", va="bottom")
    axes[0].annotate("unadjudicated band", xy=(math.sqrt(band[0] * band[1]),
                                               -0.56), color=MUTED, fontsize=7,
                     ha="center", va="bottom")
    axes[-1].set_xlabel(r"whitened counterfactual displacement  $m_{\mathrm{total}}$  (log scale)")
    handles = [plt.Line2D([], [], color=MUTED, marker="o", ms=6, ls="none",
                          mfc="none", mew=1.3, label="random encoder (null)")]
    handles += [plt.Line2D([], [], color=ARM_STYLE[a][0], marker=ARM_STYLE[a][1],
                           ms=6, ls="none", mec=SURFACE, mew=1.2,
                           label=ARM_STYLE[a][2])
                for a in ("control_strong", "color_strong", "position_strong")]
    handles.append(plt.Line2D([], [], color=ORANGE, marker="s", ms=5.5, ls="none",
                              mfc="none", mew=1.3,
                              label="color strong, var_fraction 0.999"))
    legend_ax.legend(handles=handles, loc="upper center", ncol=2,
                     handletextpad=0.6, columnspacing=1.8, borderpad=0)
    fig.suptitle("Counterfactual displacement against the calibrated resolution, "
                 "12 seeds", x=0.01, ha="left", fontsize=9.5,
                 fontweight="bold", color=INK)
    _stamp(fig, False,
           f"small dots: per-seed values · shaded: unadjudicated band "
           f"[{band[0]:.4f}, {band[1]:.4f}] · rule: m* = {m_star:.4f} · "
           f"verdicts at var_fraction 0.99\nsource: {disp_dir}/")
    foot = 0.30 / fig.get_figheight()
    fig.tight_layout(rect=(0, foot, 1, 0.955))
    _save(fig, out, "displacement_certificate", rows_csv)




def render_root(root: str | Path, out: str | Path, n_boot: int = N_BOOT) -> list[str]:
    _style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [load_cell(d) for d in discover_cells(root)]
    if not cells:
        print(f"[figures] no cells under {root} — nothing to render")
        return []
    import json
    synthetic = any("synthetic_world" in json.loads((c.path / "meta.json").read_text())
                    for c in cells)
    if synthetic:
        (out / "README.md").write_text(
            "# SYNTHETIC figures — pipeline validation only\n\nRendered from "
            "hand-set fixtures (src/probes/synthetic.py); not empirical results.\n")
    all_stats = {c.name: _cell_stats(c, n_boot) for c in cells}
    for c in cells:
        fig_g_ladder(c, all_stats[c.name], out, synthetic)
        fig_h4(c, all_stats[c.name], out, synthetic)
    fig_headline(cells, all_stats, out, synthetic)
    study = assemble([analyze_cell(c, n_boot=n_boot) for c in cells])
    fig_verdicts(study, out, synthetic)
    return sorted(p.name for p in out.glob("*.png"))


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results/probes", help="directory holding cell dirs")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/figures/<root basename>)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)



    ap.add_argument("--kappa-json", default=None,
                    help="results/hypotheses/kappa_ci.json; renders kappa_dissociation")
    ap.add_argument("--displacement-dir", default=None,
                    help="results/displacement; renders displacement_certificate")
    ap.add_argument("--claims-only", action="store_true",
                    help="skip the cell-derived figures and render only the claim figures")
    ap.add_argument("--audit-figures", action="store_true",
                    help="render the five audit figures (ladder floors, deficit plane, "
                         "selectivity, displacement rank, gate audit)")
    ap.add_argument("--scale-companion",
                    default="results/displacement/scale_companion.json",
                    help="displacement rank-stability companion (src.eval.displacement_scale)")
    args = ap.parse_args()
    out = Path(args.out or Path("results/figures") / Path(args.root).name)
    made: list[str] = []
    if args.claims_only:
        _style()
        out.mkdir(parents=True, exist_ok=True)
    else:
        made = render_root(args.root, out, n_boot=args.n_boot)
    if args.kappa_json:
        fig_kappa(Path(args.kappa_json), out)
    else:
        print("[figures] kappa_dissociation not requested (pass --kappa-json)")
    if args.displacement_dir:
        fig_displacement(Path(args.displacement_dir), out)
    else:
        print("[figures] displacement_certificate not requested "
              "(pass --displacement-dir)")
    if args.audit_figures:
        fig_ladder_floors(out)
        fig_deficit_plane(out)
        fig_selectivity(out)
        fig_gate_audit(out)
        fig_displacement_rank(Path(args.scale_companion), out)
    print(f"[figures] {len(sorted(out.glob('*.png')))} figure(s) in {out}")









PINNED = Path("results/probes_pinned")
EXCLUDED = Path("results/probes_excluded")
RUNG_LABELS = ("linear", "mlp\nsmall", "mlp\nlarge", "mlp\ndeep")


def _stack(cell: str, root: Path = PINNED):
    """(stacks, factor names, rung names) for one cell, asserting the cell exists."""
    d = root / cell
    if not (d / "stacks.npz").exists():
        raise FileNotFoundError(f"no stacks.npz for {cell} under {root}")
    z = np.load(d / "stacks.npz")
    meta = json.loads((d / "meta.json").read_text())
    return z, [f["name"] for f in meta["factors"]], meta["rungs"], meta


def _mean_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 20260906):
    """Seed-bootstrap mean and 95% interval along axis 0."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=(n_boot, x.shape[0]))
    b = x[idx].mean(1)
    return x.mean(0), np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def fig_ladder_floors(out: Path) -> None:
    """F1: recoverability against probe capacity, with the untrained floor.

    The curve behind every kappa and every epsilon-invariance verdict. Shows the
    headroom ceiling, the colour arm's negative R^2, and the closing-versus-widening
    geometry in one panel instead of in three paragraphs of numbers.
    """
    rows: list[dict] = []
    groups = [("Shapes3D", ("control_strong", "color_strong"),
               ("floor_hue", "wall_hue", "object_hue")),
              ("dSprites", ("position_strong",), ("pos_x", "pos_y"))]
    ncol = max(len(g[2]) for g in groups)
    fig, axes = plt.subplots(len(groups), ncol, figsize=(2.45 * ncol, 2.35 * len(groups)),
                             squeeze=False)
    x = np.arange(4)
    for gi, (ds, cells, factors) in enumerate(groups):
        for ci in range(ncol):
            ax = axes[gi][ci]
            if ci >= len(factors):
                ax.axis("off")
                continue
            fac = factors[ci]
            floor_drawn = False
            for cell in cells:
                z, names, _, _ = _stack(cell)
                if fac not in names:
                    raise KeyError(f"{cell} has no factor {fac}")
                i = names.index(fac)
                if not floor_drawn:
                    m, lo, hi = _mean_ci(z["random"][:, i, :])
                    ax.fill_between(x, lo, hi, color=MUTED, alpha=0.18, lw=0)
                    ax.plot(x, m, color=MUTED, lw=1.6, ls="--", zorder=2,
                            label="untrained floor")
                    floor_drawn = True
                    for r in range(4):
                        rows.append(dict(dataset=ds, factor=fac, cell="random",
                                         rung=r, mean=float(m[r]),
                                         lo=float(lo[r]), hi=float(hi[r])))
                col, mk, lab = ARM_STYLE[cell]
                m, lo, hi = _mean_ci(z["trained"][:, i, :])
                ax.fill_between(x, lo, hi, color=col, alpha=0.20, lw=0)
                ax.plot(x, m, color=col, marker=mk, ms=4.5, lw=2, mec=SURFACE, mew=1.2,
                        label=lab, zorder=3)
                for r in range(4):
                    rows.append(dict(dataset=ds, factor=fac, cell=cell, rung=r,
                                     mean=float(m[r]), lo=float(lo[r]), hi=float(hi[r])))
            ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
            ax.set_xticks(x); ax.set_xticklabels(RUNG_LABELS)
            ax.set_title(f"{ds} · {fac}", loc="left")
            ax.set_ylim(-0.42, 1.06)
            if ci == 0:
                ax.set_ylabel("recoverability")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(0.02, 0.42),
                       ncol=1, fontsize=6.5)
    fig.suptitle("Recoverability against probe capacity, with the untrained-encoder floor",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _stamp(fig, False, "12 seeds · bands: 95% seed bootstrap · floor is one shared "
                       "untrained array per dataset · source: results/probes_pinned/")
    _save(fig, out, "ladder_floors", rows)





HEADLINE_READOUTS = {
    "color_strong": ("floor_hue", "wall_hue", "object_hue"),
    "control_strong": ("floor_hue", "wall_hue", "object_hue"),
    "color_weak": (),
    "position_strong": ("pos_x", "pos_y"),
}
LABEL_NUDGE = {
    ("color_strong", "floor_hue"): (9, -11),
    ("color_strong", "wall_hue"): (9, 1),
    ("color_strong", "object_hue"): (-20, 11),
    ("control_strong", "floor_hue"): (9, 1),
    ("control_strong", "wall_hue"): (-6, -13),
    ("control_strong", "object_hue"): (9, -10),
    ("position_strong", "pos_x"): (9, 4),
    ("position_strong", "pos_y"): (9, -12),
}


def fig_deficit_plane(out: Path) -> None:
    """F2: the deficit plane, which is what the closure fraction is a ratio of.

    kappa = 1 - D(top)/D(linear) hides both of its arguments and is undefined or
    unstable as D(linear) -> 0. Plotting the pair shows the regime directly: below
    the diagonal is closure, above it is widening, and the undefined readouts are a
    half-plane rather than a gutter.
    """
    rows: list[dict] = []
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    lim = 1.35
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=ORANGE, alpha=0.07, lw=0)
    ax.fill_between([0, lim], [0, 0], [0, lim], color=BLUE, alpha=0.07, lw=0)
    ax.plot([0, lim], [0, lim], color=BASELINE, lw=1.2, ls="--", zorder=2)
    ax.axvline(0, color=BASELINE, lw=1.0)
    ax.axhline(0, color=BASELINE, lw=1.0)
    for cell in ("control_strong", "color_strong", "color_weak", "position_strong"):
        z, names, _, _ = _stack(cell)
        col, mk, lab = ARM_STYLE[cell]
        D = z["random"] - z["trained"]
        tgt = HEADLINE_READOUTS[cell]
        for i, f in enumerate(names):
            dl, dt = float(D[:, i, 0].mean()), float(D[:, i, -1].mean())
            undef = dl <= 0
            kap = float("nan") if undef else 1 - dt / dl
            rows.append(dict(cell=cell, factor=f, D_linear=dl, D_top=dt, kappa=kap,
                             kappa_defined=not undef,
                             targeted=f in TARGETED_READOUTS[cell],
                             highlighted=f in tgt))
            ax.scatter(dl, dt, s=58 if f in tgt else 30, marker=mk,
                       facecolor=col if f in tgt else "none", edgecolor=col,
                       lw=1.6, zorder=4, alpha=0.95)
            if f in tgt:
                dx, dy = LABEL_NUDGE.get((cell, f), (7, 4))
                ax.annotate(f.replace("_", " "), (dl, dt), textcoords="offset points",
                            xytext=(dx, dy), fontsize=6.2, color=INK2)
    ax.axvspan(-0.62, 0.0, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.text(-0.58, 1.28, "κ undefined\n(D linear ≤ 0)", color=MUTED, fontsize=6.5,
            va="top")
    ax.text(0.04, 1.28, "deficit WIDENS with capacity  (κ < 0)", color=ORANGE,
            fontsize=7.5, fontweight="bold", va="top")
    ax.text(0.50, 0.20, "deficit CLOSES  (κ > 0)", color=BLUE,
            fontsize=7.5, fontweight="bold")
    ax.set_xlim(-0.62, lim); ax.set_ylim(-0.62, 1.42)
    ax.set_xlabel("linear-rung deficit  D(linear) = R(untrained) − R(trained)")
    ax.set_ylabel("top-rung deficit  D(deep)")
    ax.set_title("The deficit plane: both arguments of the closure fraction",
                 loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(True, axis="both")
    handles = [plt.Line2D([], [], ls="", marker=ARM_STYLE[c][1], color=ARM_STYLE[c][0],
                          label=ARM_STYLE[c][2]) for c in
               ("control_strong", "color_strong", "color_weak", "position_strong")]
    handles.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec=MUTED,
                              label="other readout (open)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.80),
              fontsize=6.2)
    fig.tight_layout()
    _stamp(fig, False, "seed means, 12 seeds · filled = the readouts the "
                       "dissociation is read on · source: results/probes_pinned/*/stacks.npz")
    _save(fig, out, "deficit_plane", rows)


def fig_selectivity(out: Path) -> None:
    """F3: the control task, on both arms and at both ends of the ladder.

    The memorization objection is about the arm where recoverability climbs, so the
    control arm is where selectivity has to be shown. A readout whose selectivity is
    near zero at the top rung is one a permuted-label probe reproduces.
    """
    rows: list[dict] = []
    cells = ("control_strong", "color_strong", "color_weak", "position_strong")
    fig, axes = plt.subplots(1, len(cells), figsize=(2.55 * len(cells), 2.9), squeeze=False)
    for ci, cell in enumerate(cells):
        ax = axes[0][ci]
        z, names, _, _ = _stack(cell)
        S = z["trained"] - z["perm"]
        tgt = TARGETED_READOUTS[cell]
        show = [f for f in names if f in tgt] or list(names)[:3]
        if cell == "control_strong":
            show = ["floor_hue", "wall_hue", "object_hue"]
        y = np.arange(len(show))
        col = ARM_STYLE[cell][0]
        for j, f in enumerate(show):
            i = names.index(f)
            for rung, off, alpha, lab in ((0, +0.16, 0.45, "linear"),
                                          (3, -0.16, 1.0, "deepest")):
                m, lo, hi = _mean_ci(S[:, i, rung])
                ax.plot([lo, hi], [y[j] + off] * 2, color=col, lw=2.4, alpha=alpha,
                        solid_capstyle="round", zorder=3)
                ax.plot([m], [y[j] + off], marker="o", ms=5, color=col, alpha=alpha,
                        mec=SURFACE, mew=1.2, zorder=4,
                        label=lab if j == 0 else None)
                rows.append(dict(cell=cell, factor=f, rung=("linear", "", "", "deep")[rung],
                                 selectivity=float(m), lo=float(lo), hi=float(hi)))
        ax.axvline(0, color=CRITICAL, lw=1.2, zorder=2)
        ax.set_yticks(y); ax.set_yticklabels(show, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(ARM_STYLE[cell][2], loc="left", fontsize=7.5)
        ax.set_xlabel("selectivity  S = R − R(permuted)")
        ax.grid(True, axis="x")
        if ci == 0:
            ax.legend(loc="lower right", fontsize=6.5)


    s3 = [axes[0][i] for i, c in enumerate(cells) if c != "position_strong"]
    hi = max(a.get_xlim()[1] for a in s3)
    lo = min(a.get_xlim()[0] for a in s3)
    for a in s3:
        a.set_xlim(lo, hi)
    fig.suptitle("Selectivity against the random-label control, both arms, both ends "
                 "of the ladder", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _stamp(fig, False, "12 seeds · 95% seed bootstrap, 4,000 draws · a readout at zero "
                       "is one the permuted-label probe reproduces")
    _save(fig, out, "selectivity", rows)


def fig_displacement_rank(comp_json: Path, out: Path) -> None:
    """F4: how the displacement statistic and its verdict move with the retained rank.

    The certificate's threshold is absolute and was calibrated on a rank-1 fixture,
    where m_total does not depend on the retained rank. On the real fields it does.
    The right panel shows the ratio to the untrained null, which is the reading that
    survives the preregistered sweep.
    """
    comp = json.loads(Path(comp_json).read_text())
    rows: list[dict] = []
    hues = ("floor_hue", "wall_hue", "object_hue")
    band = comp["cells"]["color_strong"]["unadjudicated_band"]
    mstar = comp["cells"]["color_strong"]["m_star"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.5))

    axL.axhspan(band[0], band[1], color=MUTED, alpha=0.16, lw=0)
    axL.axhline(mstar, color=INK, lw=1.3)
    axL.text(0.02, mstar * 1.12, "m*", transform=axL.get_yaxis_transform(),
             fontsize=7, color=INK)
    axL.text(0.02, band[1] * 0.55, "unadjudicated band", fontsize=6.5, color=MUTED,
             transform=axL.get_yaxis_transform())
    for f, mk in zip(hues, ("o", "s", "^")):
        d = comp["rank_scaling"][f"trained/{f}"]
        axL.plot(d["rank"], d["m_total"], marker=mk, ms=5, lw=1.8, color=ORANGE,
                 mec=SURFACE, mew=1.2, label=f"{f} (α={d['exponent_alpha']:+.2f})")
        dn = comp["rank_scaling"][f"random/{f}"]
        axL.plot(dn["rank"], dn["m_total"], marker=mk, ms=4.5, lw=1.5, ls="--",
                 color=MUTED, mec=SURFACE, mew=1.0, alpha=0.9)
        for k, m, vf in zip(d["rank"], d["m_total"], d["var_fractions"]):
            rows.append(dict(role="trained", factor=f, var_fraction=vf, rank=k,
                             m_total=m, exponent_alpha=d["exponent_alpha"],
                             ratio_to_null="", null_ratio_spread=""))
        for k, m, vf in zip(dn["rank"], dn["m_total"], dn["var_fractions"]):
            rows.append(dict(role="random", factor=f, var_fraction=vf, rank=k,
                             m_total=m, exponent_alpha=dn["exponent_alpha"],
                             ratio_to_null="", null_ratio_spread=""))
    axL.set_yscale("log"); axL.set_xscale("log")
    axL.set_xlabel("retained whitening rank k  (var. fraction 0.95 → 0.99 → 0.999)")
    axL.set_ylabel(r"$m_{\mathrm{total}}$")
    axL.set_title("Absolute scale moves with the rank  ·  calibration assumes α = 0",
                  loc="left", fontsize=8.5)
    handL, _ = axL.get_legend_handles_labels()
    handL.append(plt.Line2D([], [], ls="--", lw=1.5, color=MUTED, label="untrained null"))
    axL.legend(handles=handL, loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=6.3)
    axL.grid(True, axis="both")



    axis_vfs = [0.95, 0.99, 0.999]
    for f, mk in zip(hues, ("o", "s", "^")):
        d = comp["null_ratio_stability"][f]
        xs = [i for i, vf in enumerate(axis_vfs) if str(vf) in d]
        vals = [d[str(axis_vfs[i])] for i in xs]
        axR.plot(xs, vals, marker=mk, ms=5, lw=1.8, color=ORANGE, mec=SURFACE, mew=1.2,
                 label=f"{f} (spread {d['spread_max_over_min']:.1f}×)")
        for i, v in zip(xs, vals):
            rows.append(dict(role="trained/null", factor=f, var_fraction=axis_vfs[i],
                             rank="", m_total="", exponent_alpha="",
                             ratio_to_null=v,
                             null_ratio_spread=d["spread_max_over_min"]))
    axR.axhline(1.0, color=BASELINE, lw=1.2, ls="--")
    axR.text(0.02, 0.72, "parity with the untrained encoder", fontsize=6.3, color=MUTED,
             transform=axR.transAxes)
    axR.set_yscale("log")
    axR.set_ylim(5e-3, 3.0)
    axR.set_xlim(-0.25, len(axis_vfs) - 0.75)
    axR.set_xticks(range(len(axis_vfs)))
    axR.set_xticklabels([str(v) for v in axis_vfs])
    axR.set_xlabel("variance fraction")
    axR.set_ylabel(r"$m_{\mathrm{total}}$(trained) / $m_{\mathrm{total}}$(untrained)")
    axR.set_title("Ratio to the null, color arm only", loc="left", fontsize=8.5)
    axR.legend(loc="upper left", bbox_to_anchor=(0.02, 0.62), fontsize=6.3)
    axR.grid(True, axis="both")
    fig.suptitle("Color-arm hue displacement against the retained whitening rank",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _stamp(fig, False, "12 seeds, seed means · preregistered var-fraction sweep · "
                       "source: results/displacement/scale_companion.json")
    _save(fig, out, "displacement_rank", rows)


PROBES = Path("results/probes")





GATE_LABEL = {
    "color_strong":        ("color, strong",                     ORANGE),
    "color_weak":          ("color, weak (exploratory)",          AQUA),
    "position_strong":     ("position, strong",                  YELLOW),
    "control_strong":      ("control-aug (minimal, exempt)",       BLUE),
    "scale_recipe_strong": ("scale + photometric base (n=1)",     AQUA),
    "orientation_strong":  ("orientation (excluded)",             MUTED),
    "scale_strong":        ("scale (excluded)",                   MUTED),
}
GATE_PANELS = (
    ("Pinned confirmatory configuration  ·  probe-train 2,000, 1,000 steps",
     [("color_strong", PINNED), ("color_weak", PINNED), ("position_strong", PINNED),
      ("control_strong", PINNED)]),
    ("Earlier configuration, where the two exclusions were decided  ·  probe-train 40,000",
     [("color_strong", PROBES), ("control_strong", PROBES), ("position_strong", PROBES),
      ("scale_recipe_strong", EXCLUDED), ("orientation_strong", EXCLUDED),
      ("scale_strong", EXCLUDED)]),
)


def fig_gate_audit(out: Path) -> None:
    """F5: the augmentation-to-factor correspondence audit, per encoder.

    Every encoder in the study on one axis: the shape anchor, a factor no condition
    targets, against its own untrained floor and the gate threshold. This is the
    measurement that excluded two designed conditions, and it runs before any
    invariance verdict is read. Panels are per probe configuration because levels
    do not transfer between them; only the ordering does.
    """
    rows: list[dict] = []
    heights = [len(p[1]) for p in GATE_PANELS]
    fig, axes = plt.subplots(len(GATE_PANELS), 1, figsize=(7.0, 4.6),
                             gridspec_kw={"height_ratios": heights})
    for ax, (title, entries) in zip(axes, GATE_PANELS):
        ax.axvspan(0.90, 1.06, color=AQUA, alpha=0.07, lw=0)
        ax.axvline(0.90, color=INK, lw=1.2, zorder=5)
        for yi, (cell, root) in enumerate(entries):
            meta = json.loads((root / cell / "meta.json").read_text())
            per = meta["quality_gate"]["per_seed"]
            lab, col = GATE_LABEL[cell]
            vals = [float(q["shape_recoverability"]) for q in per]
            floors = [float(q["floor"]) for q in per]
            ax.plot(vals, [yi] * len(vals), ls="", marker="o", ms=5.5, color=col,
                    mec=SURFACE, mew=1.1, alpha=0.85, zorder=4)
            ax.plot([float(np.mean(floors))], [yi], ls="", marker="|", ms=14,
                    color=MUTED, mew=2.2, zorder=3)
            n_pass = sum(bool(q["passed"]) for q in per)
            ax.text(1.075, yi, f"{n_pass}/{len(per)}", fontsize=6.8, va="center",
                    ha="left", color=INK2)
            for q in per:
                rows.append(dict(panel=title.split("·")[0].strip(), cell=cell,
                                 seed=q["seed_index"],
                                 shape_recoverability=float(q["shape_recoverability"]),
                                 untrained_floor=float(q["floor"]),
                                 passed=bool(q["passed"]),
                                 probe_train=meta.get("probe_train_size"),
                                 probe_steps=meta.get("probe_steps")))
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([GATE_LABEL[c][0] for c, _ in entries], fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlim(0.30, 1.13)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylim(len(entries) - 0.45, -0.55)
        ax.set_title(title, loc="left", fontsize=7.2, color=INK2)
        ax.grid(True, axis="x")
    axes[-1].set_xlabel("linear-rung shape recoverability (the preserved anchor, "
                        "targeted by no condition)")


    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=MUTED, mec=SURFACE,
                          label="one encoder"),
               plt.Line2D([], [], ls="", marker="|", ms=12, color=MUTED, mew=2.2,
                          label="untrained floor"),
               plt.Line2D([], [], color=INK, lw=1.2, label="gate: shape ≥ 0.90")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.930),
               ncol=3, fontsize=6.5)
    fig.suptitle("Correspondence audit: does an augmentation disturb an untargeted factor?",
                 x=0.01, y=0.985, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _stamp(fig, False, "pass counts at right · levels are not comparable between "
                       "panels, only the ordering is · source: results/probes_pinned/, "
                       "results/probes/, results/probes_excluded/")
    _save(fig, out, "gate_audit", rows)


if __name__ == "__main__":
    _main()
