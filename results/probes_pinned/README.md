# Pinned-config probe stacks

One folder per realized cell of the A7(d) three-cell grid. Every stack here was swept at
the A14(g) pin: `regime=interpolation`, `probe_train=2000`, `probe_steps=1000`, 12 seeds.

    color_strong/     shapes3d  color     complete
    control_strong/   shapes3d  control   complete
    position_strong/  dsprites  position  complete

`results/probes/` holds the superseded pre-A10 stacks (`probe_train=40000`, no
`probe_steps`, no `encoder_ckpts`, no `random_projector`). A10(a) voids every continuous
entry in them; they are kept for provenance only and must not feed the statistics layer.

Each cell carries `stacks.npz` with `trained`, `random`, `perm`, `projector` and
`random_projector`, all `[12, F, R]`, plus `meta.json` with per-encoder sha256 provenance
and the quality-gate result.

## Reading these from the pipeline

Notebook 00 stages 3-5 and `src.probes.hypotheses` default to `results/probes`. Point them
here explicitly:

    python -m src.probes.run_sweep ... --out-root results/probes_pinned
    python -m src.probes.hypotheses --root results/probes_pinned
