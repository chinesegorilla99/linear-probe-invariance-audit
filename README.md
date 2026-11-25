# What a Linear-Probe Verdict Cannot Tell Apart

Code and results for the paper *What a Linear-Probe Verdict Cannot Tell Apart: A Measurement Audit of Self-Supervised Invariance*. We compare linear and MLP probes with paired counterfactual measurements on Shapes3D and dSprites.

Read the [preprint on Zenodo](https://doi.org/10.5281/zenodo.24517813).

## Check the reported results

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
make manifest-check
make trace
make paper-figures
make test
```

`results/` contains the released measurements. Use `results/probes_pinned/` for the paper's analysis. The [preregistration](preregistration/prereg.md) includes the dated amendments.

Encoder checkpoints are required for new probe sweeps. The dataset commands and training entry point are:

```sh
python -m src.data.shapes3d --download --build-cache
python -m src.data.dsprites --download --build-cache
python -m src.encoders.train_simclr --config configs/run/color_strong.yaml --set run.seed=0
```

See [configs](configs/) for the other conditions and [notebooks](notebooks/) for the cloud workflow.
