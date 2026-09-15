#!/usr/bin/env python3
"""Run the frozen second-representation study. All outcomes are retained.

Example: python scripts/run_trees.py --protocol protocol_trees.json
A smoke run overrides --runs and --output and is not included in the paper.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import numba
from src.boolean_core import circuit_schema, input_tables, make_tasks
from src.tree_improver import SEED_TREE, develop_tree, tree_repair


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocol_trees.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/trees')
    parser.add_argument('--runs', type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text())
    n = config['development_runs_per_family'] if args.runs is None else args.runs
    if n < 2:
        raise ValueError('At least two independent development runs are required')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'raw.npz').exists():
        raise FileExistsError('Refusing to overwrite an existing experiment')
    families = config['families']
    cs = np.asarray(config['development_checkpoints'], dtype=np.int64)
    bs = np.asarray(config['audit_budgets'], dtype=np.int64)
    if np.any(np.diff(cs) <= 0) or np.any(cs < 1) or np.any(bs < 1):
        raise ValueError('Invalid budgets')
    cards, tags = circuit_schema(config['gates'])
    inputs = input_tables()
    tr = config['training_tasks']; au = config['audit_tasks']
    bd = config['training_budget']
    # Shape: family, independent run, regime, checkpoint, audit budget.
    scores = np.zeros((len(families), n, 4, len(cs), len(bs)), dtype=np.uint16)
    train = np.zeros((len(families), n, 3, len(cs)), dtype=np.uint16)
    policies = np.zeros((len(families), n, 3, len(cs), 10), dtype=np.uint8)
    accepted = np.zeros_like(train)
    diagnostics = np.zeros((*train.shape, 3), dtype=np.uint16)
    source_paths = ['src/boolean_core.py', 'src/tree_improver.py', 'scripts/run_trees.py']
    sources = {p: source_hash(ROOT / p) for p in source_paths}
    protocol_hash = source_hash(args.protocol)
    start = time.perf_counter()
    for fi, family in enumerate(families):
        for i in range(n):
            # Independent streams are not affected by iteration order or method.
            streams = np.random.SeedSequence([config['master_seed'], fi, i]).spawn(5)
            rng_tr, rng_au, rng_task, rng_meta, rng_audit = [np.random.default_rng(s) for s in streams]
            parents, targets = make_tasks(rng_tr, tr, config['gates'], family)
            audit_parents, audit_targets = make_tasks(rng_au, au, config['gates'], family)
            ut = rng_task.random((tr, bd, 2))
            um = rng_meta.random((cs[-1], 2))
            ur = rng_meta.random((cs[-1], 10))
            ua = rng_audit.random((au, max(bs), 2))
            seed_scores = tree_repair(SEED_TREE, audit_parents, audit_targets,
                                      cards, tags, inputs, ua, bs).sum(axis=0)
            for ci in range(len(cs)):
                scores[fi, i, 3, ci] = seed_scores
            for regime in range(3):
                p, s, a, diag = develop_tree(regime, parents, targets, cards, tags,
                                             inputs, ut, um, ur, cs, bd)
                train[fi, i, regime] = s
                policies[fi, i, regime] = p
                accepted[fi, i, regime] = a
                diagnostics[fi, i, regime] = diag
                for ci in range(len(cs)):
                    scores[fi, i, regime, ci] = tree_repair(
                        p[ci], audit_parents, audit_targets, cards, tags, inputs,
                        ua, bs).sum(axis=0)
            if (i + 1) % 128 == 0 or i == n - 1:
                print(f'{family}: {i + 1}/{n}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    elapsed = time.perf_counter() - start
    np.savez_compressed(out / 'raw.npz', scores=scores, train=train, policies=policies,
                        accepted=accepted, diagnostics=diagnostics, cs=cs, bs=bs)
    primary_ci = list(cs).index(config['primary_C'])
    primary_bi = list(bs).index(config['primary_B'])
    with (out / 'primary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['family', 'run', 'recursive', 'fixed_author', 'random_search', 'seed'])
        for fi, family in enumerate(families):
            for i in range(n):
                writer.writerow([family, i, *scores[fi, i, :, primary_ci, primary_bi].tolist()])
    metadata = {
        'protocol_sha256': protocol_hash, 'source_sha256': sources,
        'executed_config': {**config, 'development_runs_per_family': n},
        'frozen_size_used': args.runs is None, 'elapsed_seconds_including_jit': elapsed,
        'python': sys.version, 'numpy': np.__version__, 'numba': numba.__version__,
        'platform': platform.platform(),
        'score_denominator': au * 64,
        'policy_evaluations_per_development_regime': int(cs[-1]) + 1,
        'circuit_evaluations_per_development_regime': (int(cs[-1])+1) * tr * (bd+1),
        'audit_circuit_evaluations_per_policy_all_B': au * sum(int(b)+1 for b in bs),
        'source_unchanged_during_run': all(source_hash(ROOT / p) == h for p, h in sources.items()),
        'raw_sha256': source_hash(out / 'raw.npz'),
        'primary_sha256': source_hash(out / 'primary.csv')
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'elapsed': elapsed, 'output': str(out), 'shape': scores.shape}), flush=True)


if __name__ == '__main__':
    main()
