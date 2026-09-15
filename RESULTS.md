# Executed results

## Conditional-tree independent confirmation

4,096 independent development seeds per family, C=64, B=64. Effects are percentage points. Each seed contains paired recursive, fixed-author, and uniform whole-program search runs; these methods are not independent replications.

| Family | Recursive gain | Fixed-author gain | Recursion dividend | Simultaneous 95% interval |
|---|---:|---:|---:|---:|
| Mixed | 3.28 | 4.78 | -1.50 | [-2.35, -0.65] |
| XOR-heavy | 2.89 | 4.23 | -1.34 | [-2.17, -0.51] |
| Logic-heavy | 3.75 | 5.41 | -1.66 | [-2.53, -0.80] |

All three primary dividend intervals exclude zero. All nine seed-relative gains, including whole-program random search, have positive lower bounds in their separate confidence family. A stricter J=12 joint-family check preserves all signs. This is a bounded-system negative inheritance effect despite positive mechanism improvement, not a claim that RSI generally fails.

The initial 2,048-seed-per-family tree study is retained separately. All six meta-type permutations were subsequently tested with another 2,048 new seed sets per family. Every point-estimated dividend was negative, but that diagnostic's 18-comparison intervals include zero. None of its 15 nonidentity adapter-effect intervals excludes zero; this does not establish equivalence.

## Retained original rule-table confirmation

At C=128, B=32, with 4,096 development seeds per family, both local methods improve over the seed. All three primary dividend intervals fit inside a prespecified +/-1 percentage-point practical-equivalence band. This is tolerance-specific equivalence, not exact equality. Whole-policy random search is stronger in mean and is not omitted.

## Evidence and verification

- `results/trees_confirmation/summary.json` and `sufficient.json`: all primary means, intervals, and exact integer sufficient statistics.
- `results/trees_confirmation/joint_family_check.json`: stricter joint-family sensitivity check.
- `results/trees/`: exploratory grid and execution metadata.
- `results/adapter_diagnostic/`: every mapping result and source/protocol/raw hashes.
- `results/confirmation/`: original confirmation, preserved.
- `results/trees_extension_tests.txt`: 70 local tests passed.

`make verify-aggregates` recomputes the new published intervals without rerunning search. Full verifiers additionally check raw data, source hashes, and deterministic seed-prefix replay. Raw arrays and per-run CSVs are included in the accompanying research archive and can be regenerated from the frozen protocols. Replaying old seeds is not counted as new independent evidence.
