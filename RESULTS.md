# Completed research results

This is a working research draft, not a submitted or independently reviewed paper.

The independent Boolean-program confirmation used 4,096 development seeds per family, 128 development proposals, and 32 downstream repair proposals. Each seed contains paired recursive, fixed-author, and random-search development trajectories. The fixed-author method may discover and retain improved policies; only its proposer remains the seed.

Effects below are percentage points of correct truth-table outputs.

| Family | Recursive gain over seed | Fixed-author gain over seed | Recursion dividend | Simultaneous finite-sample 95% interval |
|---|---:|---:|---:|---:|
| Mixed | 5.55 | 5.61 | -0.065 | [-0.841, 0.711] |
| XOR-heavy | 4.08 | 4.10 | -0.021 | [-0.771, 0.730] |
| Logic-heavy | 6.21 | 6.15 | 0.058 | [-0.725, 0.842] |

All three primary intervals lie within the prespecified one-percentage-point practical-equivalence band. This does not establish exact equality or a general impossibility of recursive self-improvement. Random policy search performs better than either local-development method, and its outcomes are retained.

The original 512-seed-per-family grid and the independent confirmation are separate. The confirmation margin and empirical Bernstein analysis were chosen after initial results but before confirmation seeds were executed. This is not an external preregistration. The nine seed-relative gains use a separate simultaneous confidence family.

A complete 17-page draft is written in the official ICLR 2027 format, with seven main-text pages and full proofs and experimental appendices. The theory includes comparator-specific causal effects, score-summary nonidentification, sharp finite-state attribution intervals under rectangular uncertainty, shared-history cancellation, a terminal-audit detection boundary, and development-level statistical certificates. Established mathematical ingredients are explicitly attributed rather than claimed as newly invented.

In the stated terminal-audit model, if b bounds total contrast distortion, uniform detection of a true effect gamma is impossible when gamma <= 2*b. Above that boundary the sample size scales as log(1/delta)/(gamma-2*b)^2. The assumptions and full proof are in the manuscript.

All 35 software tests pass. Saved seed prefixes replay bitwise exactly for all three families in both studies, and every confirmation interval recomputes from the saved integer scores. Measured CPU execution was about 59 seconds for the initial suite and 314 seconds for confirmation in the recorded environment; no GPU or LLM API was used.

Before submission, the paper still needs human-author proof/code verification, a critical novelty review, and preferably a second executable improver representation. The public repository is identifying and must not be used as an identifying link in an anonymous submission.
