.PHONY: test experiments confirmation analyze figures paper verify

test:
	python -m pytest -q
experiments:
	python scripts/run_experiments.py --suite all
confirmation:
	python scripts/run_experiments.py --suite programs --protocol protocol_confirmation.json --out results/confirmation
analyze:
	python scripts/analyze.py
figures:
	python scripts/plot_results.py
paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
verify:
	python scripts/verify_artifacts.py
