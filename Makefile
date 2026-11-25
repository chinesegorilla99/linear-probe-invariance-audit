PY ?= python

.PHONY: test hypotheses figures paper-figures kaggle-bundle kaggle-bundle-check trace pixel-shapes3d pixel-dsprites manifest manifest-check paper paper-clean submission submission-check

test:
	$(PY) -m unittest discover -s tests

hypotheses:
	$(PY) -m src.probes.hypotheses --root results/probes

figures:
	$(PY) -m src.eval.figures --root results/probes

PAPER_FIGURES = gate_audit.pdf ladder_floors.pdf kappa_dissociation.pdf \
	deficit_plane.pdf selectivity.pdf displacement_certificate.pdf \
	displacement_rank.pdf

paper-figures:
	$(PY) -m src.eval.figures --root results/probes_pinned --out results/figures/real \
		--kappa-json results/hypotheses/kappa_ci.json \
		--displacement-dir results/displacement \
		--audit-figures
	cp $(PAPER_FIGURES:%=results/figures/real/%) paper/figures/

pixel-shapes3d:
	$(PY) -m src.probes.pixel_baseline --dataset shapes3d

pixel-dsprites:
	$(PY) -m src.probes.pixel_baseline --dataset dsprites

manifest:
	$(PY) -m src.eval.manifest

manifest-check:
	$(PY) -m src.eval.manifest --check

kaggle-bundle:
	$(PY) -m src.eval.kaggle_bundle

kaggle-bundle-check:
	$(PY) -m src.eval.kaggle_bundle --check

trace:
	$(PY) -m src.eval.trace_numbers

TEXBIN ?= /usr/local/texlive/bin

paper:
	cd paper && PATH="$$PATH:$(TEXBIN)" latexmk -pdf main_rewritten.tex

paper-clean:
	cd paper && PATH="$$PATH:$(TEXBIN)" latexmk -C main_rewritten.tex

submission:
	cd paper && rm -f tmlr-submission.zip && zip -q tmlr-submission.zip \
		main_rewritten.tex references.bib tmlr.sty tmlr.bst fancyhdr.sty \
		$$(grep -o 'figures/[A-Za-z0-9_-]*\.pdf' main_rewritten.tex | sort -u)
	unzip -l paper/tmlr-submission.zip

submission-check:
	rm -rf paper/build/submission-check && mkdir -p paper/build/submission-check
	unzip -q paper/tmlr-submission.zip -d paper/build/submission-check
	cd paper/build/submission-check && PATH="$$PATH:$(TEXBIN)" \
		latexmk -pdf -interaction=nonstopmode -file-line-error main_rewritten.tex
