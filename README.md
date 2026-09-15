# The Causal Recursion Dividend

A theory-first, CPU-only investigation of **whether an improver became better** and **whether recursively inheriting that improver helped**.

## New: independently confirmed negative inheritance dividend

The conditional-tree extension has 4,096 new development seeds per family at C=64 and B=64. Effects below are percentage points on fresh Boolean-repair tasks.

| Family | Recursive gain over seed | Fixed-author gain over seed | Recursion dividend | Simultaneous finite-sample 95% interval |
|---|---:|---:|---:|---:|
| Mixed | 3.28 | 4.78 | -1.50 | [-2.35, -0.65] |
| XOR-heavy | 2.89 | 4.23 | -1.34 | [-2.17, -0.51] |
| Logic-heavy | 3.75 | 5.41 | -1.66 | [-2.53, -0.80] |

Both local methods learn better improvers, but recursive inheritance is worse under this declared comparison. All three dividend intervals exclude zero. Whole-tree random search is stronger in mean and is retained. The exploratory 2,048-seed study is separate; no samples were pooled into confirmation.

A further 2,048-seed-per-family diagnostic tests all six task-to-meta type permutations. It finds no detected remapping benefit; its conservative intervals do not establish equivalence. Complete results, including these unhelpful diagnostic outcomes, are retained. This is a second representation within the same Boolean domain, not evidence about general-purpose AI.

A stricter joint J=12 confidence-family check also preserves all three negative dividends and all nine positive seed gains.

The revision has an eight-page main text and 23 pages including complete appendices, with 70 passing local tests. See `notes/RESEARCH_STATUS.md` for remaining independent-review and novelty gates.

## Original rule-table result

A complete ICLR-2027-format research draft, full proofs, executable experiments, and independent confirmation are included. This is **not a submitted or independently reviewed paper**. No broad autonomous-RSI claim is made.

The independent confirmation used 4,096 development seeds per family, 128 development proposals, and 32 downstream repair proposals. Each seed produces paired recursive, fixed-author, and random-search development trajectories. Scores are fractions of correct Boolean truth-table outputs; the table below reports **percentage points**.

| Task family | Recursive gain over seed | Fixed-author gain over seed | Recursion dividend | Simultaneous finite-sample 95% interval |
|---|---:|---:|---:|---:|
| Mixed | 5.55 | 5.61 | -0.065 | [-0.841, 0.711] |
| XOR-heavy | 4.08 | 4.10 | -0.021 | [-0.771, 0.730] |
| Logic-heavy | 6.21 | 6.15 | 0.058 | [-0.725, 0.842] |

All three dividend intervals lie inside the **prespecified +/-1 percentage-point equivalence band**. Both development procedures acquire better improvers, but recursive inheritance provides little additional benefit in this bounded rule-table benchmark. This is practical equivalence at the stated tolerance, not exact equality or a general impossibility of RSI. Random policy search performs better than both local-development methods; those results are retained.

The three primary intervals use the Maurer--Pontil empirical Bernstein bound with a familywise correction. Nine seed-relative gains have a **separate** simultaneous 95% confidence family. The initial 512-seed-per-family sweep is retained separately. The confirmation protocol and tolerance were fixed after initial results but before independent confirmation seeds were executed; neither study is represented as an external preregistration.

## Theory

The manuscript defines transferable mechanism gain and a comparator-specific causal recursion dividend. It proves native-summary nonidentification, sharp finite-state attribution intervals under rectangular transition uncertainty, shared-history cancellation, an endpoint-audit minimax boundary, development-level certificates, and a nested-replication lower bound.

In the endpoint model, `b` bounds total contrast distortion. Uniformly detecting a true effect `gamma` is impossible when `gamma <= 2*b`; above the boundary the required independent-pair count is of order `log(1/delta)/(gamma-2*b)^2`. The distinction between a positive observed certificate and uniform power is essential.

A new persistent-adapter proposition shows why a fixed shared interface cannot be replaced by independent rowwise uncertainty: the same two-step system has sharp persistent interval [0.2,0.2] but rowwise relaxation [-0.2,0.2]. A separate proof establishes pointwise fixed-author invariance in the type-permutation diagnostic. Robust Bellman recursion, performance-difference identities, empirical Bernstein concentration, and nested-variance formulas are established tools and are attributed as such. See `notes/RESEARCH_STATUS.md` for the novelty and validation assessment.

## Reproduce

Use Python 3.13 and the recorded dependency versions:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_experiments.py --suite all
python scripts/run_experiments.py --suite programs --protocol protocol_confirmation.json --out results/confirmation
python scripts/analyze.py
python scripts/run_trees.py --protocol protocol_trees.json --output results/trees
python scripts/run_trees.py --protocol protocol_trees_confirmation.json --output results/trees_confirmation
python scripts/analyze_trees.py --results results/trees
python scripts/analyze_trees.py --results results/trees_confirmation
python scripts/run_adapter_diagnostic.py
python scripts/analyze_adapter_diagnostic.py
python scripts/plot_results.py
python scripts/plot_trees.py
python scripts/verify_artifacts.py
python scripts/verify_trees.py
python scripts/verify_adapter_diagnostic.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

To build the manuscript from the aggregate data already on GitHub, run `make figures paper` (requires the Python dependencies and a LaTeX installation). This does not require rerunning the search experiments. `make verify-aggregates` checks the new confidence bounds from tracked integer sufficient statistics without raw arrays. Full verification additionally needs the raw archives from the research supplement or a fresh run. The new runners refuse to overwrite existing raw outputs.

For a quick software check, use `python scripts/run_experiments.py --smoke`; smoke outputs are excluded from paper estimates. Measured single-process execution was about 59 seconds for the initial suite and 314 seconds for confirmation in the recorded environment, not a runtime promise for another machine. The conditional-tree exploration, independent confirmation, and full mapping diagnostic took approximately 156, 260, and 253 seconds respectively. Their protocols were fixed before their own outcomes, not externally preregistered.

## Layout and data

- `paper/main.tex`, `paper/proofs.tex`, `paper/experiment_details.tex`: manuscript and complete assumptions/proofs.
- `src/theory.py`: TV optimization, robust dynamic programming, exact decomposition, statistical intervals.
- `src/programs.py`: original protected Boolean interpreter and bounded self-modifying rule table, preserved unchanged.
- `src/tree_improver.py`, `src/boolean_core.py`: typed conditional-program improvers and checked task-core extraction.
- `src/shared_adapter.py`, `src/adapter_diagnostic.py`: exact persistent-interface models and controlled meta-type interventions.
- `scripts/`: experiment runner, raw-result analysis, plots, artifact checks.
- `protocol*.json`: separate initial and confirmation specifications.
- `results/`: selected aggregate findings, experiment metadata, and validation reports. The full initial grid, all resource logs, and raw outcomes are included in the research archive.

Raw NPZ arrays and per-development-run CSVs are provided in the full research archive accompanying the working draft and are regenerated by the commands above. The repository tracks selected aggregate results, exact integer sufficient statistics, and source; the verification report records the full archive's raw-data checksums. Generated PDF/figure binaries are also reproducible from source. Do not interpret omitted generated binaries as unexecuted experiments.

The official style file is copied byte-for-byte from `ICLR/Master-Template/iclr2027/iclr2027_conference.sty` (Git blob `f61ad7efce0855557694078c0945e6c33feb8236`). The working PDF overrides only its draft label and does not claim to be under review. The manuscript includes an AI-use disclosure. Human verification and a critical novelty review remain necessary before submission. The public repository identifies its owner and must not be used as an identifying link in an anonymous submission.
