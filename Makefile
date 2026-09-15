.PHONY: test experiments confirmation trees tree-confirmation adapter-diagnostic analyze figures paper verify verify-aggregates

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
	python scripts/plot_trees.py
paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
verify:
	python scripts/verify_artifacts.py
	python scripts/verify_trees.py
	python scripts/verify_adapter_diagnostic.py

trees:
	python scripts/run_trees.py --protocol protocol_trees.json --output results/trees
	python scripts/analyze_trees.py --results results/trees
tree-confirmation:
	python scripts/run_trees.py --protocol protocol_trees_confirmation.json --output results/trees_confirmation
	python scripts/analyze_trees.py --results results/trees_confirmation
adapter-diagnostic:
	python scripts/run_adapter_diagnostic.py
	python scripts/analyze_adapter_diagnostic.py
verify-aggregates:
	python scripts/verify_trees.py --aggregate-only
	python scripts/verify_adapter_diagnostic.py --aggregate-only
