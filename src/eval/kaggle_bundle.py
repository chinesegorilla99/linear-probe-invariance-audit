"""Stage every artifact notebook 10 needs into one flat, self-describing folder.

Kaggle datasets arrive in one of two shapes: a zip upload keeps its directory tree,
loose file uploads land flat in ``/kaggle/input/<slug>/``. Nested names collide the
moment they are flattened (24 files called ``backbone.pt``, four called
``stacks.npz``), so this writes each file under its repo-relative path with ``/``
replaced by ``__``. The encoding is reversible because no source path contains a
double underscore, which ``build`` asserts rather than assumes.

Two normalizations happen on the way in. Checkpoints downloaded from Kaggle arrive as
``backbone (18).pt``; every one is renamed to the canonical ``backbone.pt`` its seed
directory should hold, which also removes the spaces. And a MANIFEST.json records each
file's destination path and sha256, so the notebook can verify what it received rather
than trusting a filename.

    python -m src.eval.kaggle_bundle                 # build kaggle_upload_nb10/
    python -m src.eval.kaggle_bundle --check         # verify an existing bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SEP = "__"
OUT = Path("kaggle_upload_nb10")

PINNED = ("color_strong", "control_strong", "color_weak", "position_strong")
PROBES = ("color_strong", "control_strong", "position_strong")
EXCLUDED = ("orientation_strong", "scale_strong", "scale_recipe_strong")


def sources() -> list[tuple[Path, str, str]]:
    """(source path, destination repo-relative path, which notebook section needs it)."""
    out: list[tuple[Path, str, str]] = []

    def add(src, dst, sec):
        out.append((Path(src), str(dst), sec))

    for c in PINNED:
        sec = "5,7" if c in ("color_strong", "control_strong") else "7"
        for f in ("stacks.npz", "meta.json"):
            add(f"results/probes_pinned/{c}/{f}", f"results/probes_pinned/{c}/{f}", sec)
    for c in PROBES:
        for f in ("stacks.npz", "meta.json"):
            add(f"results/probes/{c}/{f}", f"results/probes/{c}/{f}", "7")
    for c in EXCLUDED:
        for f in ("stacks.npz", "meta.json"):
            add(f"results/probes_excluded/{c}/{f}", f"results/probes_excluded/{c}/{f}", "7")
    for f in ("kappa_ci.json", "hypothesis_report.json"):
        add(f"results/hypotheses/{f}", f"results/hypotheses/{f}", "7")
    for p in sorted(Path("results/displacement").glob("*.json")):
        sec = "6" if p.name == "threshold_calibration.json" else "7"
        add(p, f"results/displacement/{p.name}", sec)


    for arm in ("control", "position"):
        for d in sorted(Path(f"results/encoders/{arm}").glob(f"{arm}_strong_seed*")):
            ck = sorted(d.glob("backbone*.pt"))
            if len(ck) != 1:
                raise SystemExit(f"{d}: expected exactly one backbone*.pt, found {len(ck)}")
            add(ck[0], f"results/encoders/{arm}/{d.name}/backbone.pt", "6")
    std = Path("results/encoders/standard_strong_seed1/standard_strong_seed1.pt")
    add(std, str(std), "6b")
    return out


def encode(dst: str) -> str:
    return dst.replace("/", SEP)


def sha256(p: Path, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blocksize), b""):
            h.update(b)
    return h.hexdigest()


def build(out: Path, force: bool) -> None:
    rows, missing = [], []
    for src, dst, sec in sources():
        if SEP in dst:
            raise SystemExit(f"{dst}: contains {SEP!r}; the encoding would not be reversible")
        if not src.exists():
            missing.append(str(src))
            continue
        rows.append((src, dst, sec))
    if missing:
        for m in missing:
            print(f"  MISSING {m}")
        raise SystemExit(f"{len(missing)} source file(s) missing; nothing written")

    out.mkdir(parents=True, exist_ok=True)
    manifest, total = [], 0
    for src, dst, sec in rows:
        flat = out / encode(dst)
        if force or not flat.exists() or flat.stat().st_size != src.stat().st_size:
            shutil.copy2(src, flat)
        size = flat.stat().st_size
        total += size
        manifest.append({"upload_name": flat.name, "destination": dst,
                         "section": sec, "bytes": size, "sha256": sha256(flat)})
        print(f"  {sec:>4s}  {flat.name}")

    (out / "MANIFEST.json").write_text(json.dumps(
        {"bundle": "notebook 10 audit response", "separator": SEP,
         "note": "upload_name decodes to destination by replacing the separator with '/'",
         "files": manifest}, indent=1))
    (out / "README.txt").write_text(
        "Upload every file in this folder as one Kaggle dataset.\n\n"
        "Names encode their destination: '__' stands for '/', so\n"
        "results__probes_pinned__color_strong__stacks.npz  ->  "
        "results/probes_pinned/color_strong/stacks.npz\n\n"
        "Notebook 10's section 4 decodes them, verifies each against MANIFEST.json,\n"
        "and rebuilds the tree. Loose files and a zipped folder both work; the\n"
        "notebook searches recursively and does not care which shape arrives.\n\n"
        f"{len(manifest)} files, {total / 1e9:.2f} GB.\n")
    by_sec: dict[str, int] = {}
    for m in manifest:
        by_sec[m["section"]] = by_sec.get(m["section"], 0) + m["bytes"]
    print(f"\nwrote {out}/ — {len(manifest)} files + MANIFEST.json + README.txt, "
          f"{total / 1e9:.2f} GB")
    for sec in sorted(by_sec):
        print(f"  section {sec:>4s}: {by_sec[sec] / 1e6:9.1f} MB")


def check(out: Path) -> None:
    man = json.loads((out / "MANIFEST.json").read_text())
    bad = 0
    for m in man["files"]:
        p = out / m["upload_name"]
        if not p.exists():
            print(f"  MISSING {m['upload_name']}"); bad += 1
        elif p.stat().st_size != m["bytes"] or sha256(p) != m["sha256"]:
            print(f"  CORRUPT {m['upload_name']}"); bad += 1
    print(f"{len(man['files']) - bad}/{len(man['files'])} files verified in {out}")
    raise SystemExit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true", help="recopy files already staged")
    ap.add_argument("--check", action="store_true", help="verify a bundle against its manifest")
    args = ap.parse_args()
    out = Path(args.out)
    check(out) if args.check else build(out, args.force)


if __name__ == "__main__":
    main()
