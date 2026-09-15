# Research status and submission gates

## Completed in this revision

The research manuscript has eight pages of main text and 23 pages including disclosure, references, complete proofs, and detailed experiments. It uses the official ICLR 2027 style and is marked as a research draft, not submitted. The PDF compiles without overfull boxes or unresolved references; rendered pages, tables, and figures were inspected.

The original rule-table experiment and its independent confirmation were rerun from the tracked code. Exact, sensitivity, and lineage CSVs agree after normalizing LF/CRLF line endings. All 15 original confirmation intervals recompute from saved integer outcomes. These deterministic replays are not additional independent scientific samples.

The new conditional-tree representation adds 6,144 exploratory development seed sets and 12,288 independent confirmation seed sets. Each contains paired recursive, fixed-author, and whole-program-random development. At C=64 and B=64, all three simultaneous finite-sample confirmation intervals for the recursion dividend are strictly negative, while both local methods improve over the seed. The random-search baseline remains stronger in mean.

An additional 6,144 independent diagnostic seed sets evaluate all six task-to-meta tag permutations, with a pointwise-invariant fixed-author comparator. Their point estimates do not support the proposed simple remapping explanation; conservative intervals do not establish equivalence. These outcomes are retained rather than omitted.

The complete local suite passes 70 tests. New tests compare whole task-repair trajectories against an independently written scalar Boolean implementation, check exact extraction of the original task generator, verify fixed-author invariance under all tag permutations, enumerate small robust-model alternatives, and check input isolation. Separate evidence verifiers recompute all new finite-sample intervals, source/protocol hashes, and deterministic seed-prefix replays.

## Mathematical additions and scope

The persistent-adapter proposition distinguishes an exact finite identified set, its sharp mixture interval, and the looser rowwise robustness relaxation. A two-step example gives [0.2,0.2] for every permissible persistent adapter but [-0.2,0.2] under illegal within-audit switching. This is an application of nonrectangular uncertainty principles, not a claimed new general robust-MDP result.

The adapter diagnostic has a complete pointwise invariance proof for the fixed-author comparator. The endpoint testing theorem, robust Bellman recursion, performance-difference identity, and empirical Bernstein construction are explicitly identified as specialized uses of established machinery. See PROOF_AUDIT.md and NOVELTY_AUDIT.md.

## Remaining submission gates

1. Independent adversarial review of the complete mathematical proofs and interpreter semantics. Automated checks and an AI-led proof audit do not constitute independent human verification.
2. A critical novelty assessment against RSI evaluation, evolutionary self-adaptation, algorithm configuration, robust testing, and causal evaluation. The new paper should be judged as an integrated audit methodology, not as a new general concentration inequality or a superior optimizer.
3. External validity remains limited: there are two representations but only one executable task domain. A second application domain would be stronger than another seed sweep.
4. Human authorship verification, final AI-use disclosure, and anonymous supplementary packaging before any actual submission. The public repository must not be linked as an anonymous artifact.

The second-representation gate is now completed. No experiment in the paper is a placeholder or an invented outcome. No claim of submission, acceptance, general autonomous RSI, or completed independent human review is made.
