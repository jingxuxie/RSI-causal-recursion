#!/usr/bin/env python3
"""Reproduce all experiments: python scripts/run_experiments.py --suite all.

The protocol is loaded before running and its SHA256 is saved with results.
--smoke writes to a separate directory and must not be treated as paper data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
from scipy.stats import t as student_t
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.theory import (evaluate, robust_value, development_decomposition,
                        hoeffding_interval)
from src.programs import (make_tasks, input_tables, circuit_schema, repair_scores,
                          develop, SEED_POLICY, truth)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(z, comparisons=1):
    z = np.asarray(z, dtype=float)
    mean, lower, upper = hoeffding_interval(z, comparisons=comparisons)
    se = z.std(ddof=1) / np.sqrt(len(z))
    radius = float(student_t.ppf(.975, len(z)-1) * se)
    return {'mean': mean, 'se': float(se), 't_lower': mean-radius,
            't_upper': mean+radius, 'hoeffding_lower': lower,
            'hoeffding_upper': upper, 'n_development': len(z)}


def run_exact(out, cfg, smoke):
    rng = np.random.default_rng(cfg['seed'])
    n = 128 if smoke else cfg['development_runs']
    c_grid = [1, 2, 4, 8, 16, 32, 64]
    b_grid = [1, 4, 16]
    seed_q = np.array([.75, .20, .04, .01])
    positive = np.array([seed_q, [.15, .35, .45, .05],
                         [.1, .1, .25, .55], [0, 0, 0, 1]])
    negative = np.array([seed_q, [.95, .049, .001, 0],
                         [.98, 0, .019, .001], [0, 0, 0, 1]])
    zero = np.tile(seed_q, (4, 1))
    p = np.array([.02, .04, .20, .30]); mu = np.array([1., 0, 0, 0])
    rows = []; decomposition_rows = []; raw = {}
    for label, q in [('positive', positive), ('negative', negative), ('zero', zero)]:
        pr = np.zeros((4, 4)); pf = np.zeros((4, 4))
        for j in range(4):
            pr[j, j] = q[j, :j+1].sum(); pr[j, j+1:] = q[j, j+1:]
            pf[j, j] = seed_q[:j+1].sum(); pf[j, j+1:] = seed_q[j+1:]
        sr = np.zeros(n, dtype=int); sf = np.zeros(n, dtype=int)
        uniforms = rng.uniform(size=(c_grid[-1], n))
        for c in range(1, c_grid[-1]+1):
            sr = (uniforms[c-1, :, None] >= np.cumsum(pr[sr], axis=1)).sum(axis=1)
            sf = (uniforms[c-1, :, None] >= np.cumsum(pf[sf], axis=1)).sum(axis=1)
            sr = np.minimum(sr, 3); sf = np.minimum(sf, 3)
            if c not in c_grid:
                continue
            raw[f'{label}_C{c}_r'] = sr.copy(); raw[f'{label}_C{c}_f'] = sf.copy()
            kr = np.tile(pr, (c, 1, 1)); kf = np.tile(pf, (c, 1, 1))
            for b in b_grid:
                g = 1-(1-p)**b
                vr = evaluate(kr, g, mu); vf = evaluate(kf, g, mu)
                identity, terms = development_decomposition(kr, kf, g, mu)
                assert abs(identity-(vr-vf)) < 1e-12
                row = {'regime': label, 'C': c, 'B': b, 'value_seed': g[0],
                       'value_recursive': vr, 'value_fixed': vf,
                       'gain_recursive': vr-g[0], 'gain_fixed': vf-g[0],
                       'gamma_exact': vr-vf, 'identity_error': abs(identity-(vr-vf))}
                row.update(summarize(g[sr]-g[sf], comparisons=63)); rows.append(row)
                if c == 16 and b == 4:
                    decomposition_rows.extend({'regime': label, 'generation': t+1,
                                               'contribution': v} for t, v in enumerate(terms))
    write_csv(out/'exact.csv', rows)
    write_csv(out/'development_decomposition.csv', decomposition_rows)
    np.savez_compressed(out/'exact_raw.npz', **raw)
    # Two systems with identical observed diagonal outcomes but opposite effects.
    worlds = {'world_I': [[.4, .7], [.5, .8]], 'world_II': [[.4, .3], [.9, .8]]}
    (out/'nonidentification.json').write_text(json.dumps(worlds, indent=2)+'\n')


def run_sensitivity(out):
    h = 8; alpha = .05
    mu = np.array([1-alpha, 0, alpha, 0]); reward = np.array([0, 1., 0, 1.])
    masks = np.zeros((4, 4), dtype=bool)
    masks[:2, :2] = True; masks[2:, 2:] = True
    kernels = []
    for p in [.06, .02]:
        k = np.zeros((4, 4))
        for base in [0, 2]:
            k[base, base:base+2] = [1-p, p]
            k[base+1, base+1] = 1.
        kernels.append(np.tile(k, (h, 1, 1)))
    nominal = evaluate(kernels[0], reward, mu)-evaluate(kernels[1], reward, mu)
    rows = []
    for eps in np.linspace(0, .2, 21):
        for location in ['rare', 'global']:
            radii = np.tile([0, 0, eps, eps] if location == 'rare' else [eps]*4, (h, 1))
            r = robust_value(kernels[0], reward, mu, radii, masks)
            f = robust_value(kernels[1], reward, mu, radii, masks)
            lower, upper = r.lower-f.upper, r.upper-f.lower
            # Check that each sharp endpoint is attained by an allowed pair.
            witness_lower = evaluate(r.lower_kernels, reward, mu)-evaluate(f.upper_kernels, reward, mu)
            witness_upper = evaluate(r.upper_kernels, reward, mu)-evaluate(f.lower_kernels, reward, mu)
            assert abs(witness_lower-lower) < 1e-12
            assert abs(witness_upper-upper) < 1e-12
            beta = 1-(1-eps)**h
            rows.append({'location': location, 'epsilon': eps, 'h': h,
                         'nominal': nominal, 'lower': lower, 'upper': upper,
                         'uniform_lower': max(-1., nominal-2*beta),
                         'uniform_upper': min(1., nominal+2*beta),
                         'lower_attainment_error': abs(witness_lower-lower),
                         'upper_attainment_error': abs(witness_upper-upper)})
    write_csv(out/'sensitivity.csv', rows)
    # Uncertain scratch-state transitions cannot alter a constant continuation.
    scratch = np.tile([[.5, .5, 0, 0], [.5, .5, 0, 0],
                       [0, 0, .5, .5], [0, 0, .5, .5]], (h, 1, 1))
    ans = robust_value(scratch, [0, 0, 1, 1], [.3, 0, .7, 0], 1, masks)
    assert abs(ans.lower-.7) < 1e-12 and abs(ans.upper-.7) < 1e-12
    (out/'scratch_invariance.json').write_text(json.dumps({'lower': ans.lower, 'upper': ans.upper})+'\n')


def run_statistics(out, cfg, smoke):
    rng = np.random.default_rng(cfg['seed']); reps = 1000 if smoke else cfg['repetitions']
    alpha = cfg['alpha']; n = 16; amplitude = .4; rows = []; raw = {}
    for m in [1, 4, 16, 64, 256, 1024]:
        signs = rng.choice([-1, 1], size=(reps, n))
        counts = rng.binomial(m, amplitude, size=(reps, n))
        cluster = signs*counts/m; estimates = cluster.mean(axis=1)
        naive_radius = np.sqrt(2*np.log(2/alpha)/(n*m))
        cluster_radius = np.sqrt(2*np.log(2/alpha)/n)
        cluster_se = cluster.std(axis=1, ddof=1)/np.sqrt(n)
        t_radius = student_t.ppf(1-alpha/2, n-1)*cluster_se
        rows.append({'n': n, 'm': m, 'repetitions': reps,
                     'variance_exact': (amplitude**2+amplitude*(1-amplitude)/m)/n,
                     'variance_empirical': float(estimates.var(ddof=1)),
                     'naive_hoeffding_coverage': float(np.mean(np.abs(estimates)<=naive_radius)),
                     'cluster_hoeffding_coverage': float(np.mean(np.abs(estimates)<=cluster_radius)),
                     'cluster_t_coverage': float(np.mean(np.abs(estimates)<=t_radius))})
        raw[f'lineage_m{m}_estimates'] = estimates
    write_csv(out/'lineage.csv', rows)
    # Least favorable worlds for the endpoint-model separation theorem.
    rows = []; b = .03
    for gamma in [.04, .08, .12]:
        if gamma <= 2*b:
            observable0 = observable1 = gamma/2
        else:
            observable0, observable1 = b, gamma-b
        for sample_n in [64, 256, 1024, 4096, 16384]:
            p0r, p0f = .5+observable0/2, .5-observable0/2
            p1r, p1f = .5+observable1/2, .5-observable1/2
            d0 = (rng.binomial(sample_n, p0r, reps)-rng.binomial(sample_n, p0f, reps))/sample_n
            d1 = (rng.binomial(sample_n, p1r, reps)-rng.binomial(sample_n, p1f, reps))/sample_n
            threshold = gamma/2
            rows.append({'b': b, 'gamma': gamma, 'n': sample_n, 'repetitions': reps,
                         'type_I': float(np.mean(d0>threshold)),
                         'type_II': float(np.mean(d1<=threshold)),
                         'uniform_upper': min(1., np.exp(-sample_n*(gamma-2*b)**2/8)) if gamma>2*b else 1.})
            raw[f'separation_g{gamma}_n{sample_n}_null'] = d0
            raw[f'separation_g{gamma}_n{sample_n}_alternative'] = d1
    write_csv(out/'separation.csv', rows)
    np.savez_compressed(out/'statistics_raw.npz', **raw)


def run_programs(out, cfg, smoke):
    n = 4 if smoke else cfg['development_runs']
    families = cfg['families']; checkpoints = np.array(cfg['development_checkpoints'], dtype=np.int64)
    budgets = np.array(cfg['audit_budgets'], dtype=np.int64)
    g = cfg['gates']; ktrain = cfg['training_tasks']; kaudit = cfg['audit_tasks']
    bdev = cfg['development_repair_budget']; cards, tags = circuit_schema(g); inputs = input_tables()
    rows = []; raw_rows = []; resources = []
    for fi, family in enumerate(families):
        started = time.perf_counter()
        # Store exact integer audit scores before any aggregation.
        scores = np.empty((n, 4, len(checkpoints), kaudit, len(budgets)), dtype=np.uint8)
        policies = np.empty((n, 3, len(checkpoints), 8), dtype=np.int64)
        dev_scores = np.empty((n, 3, len(checkpoints)), dtype=np.int64)
        acceptances = np.empty_like(dev_scores)
        initial_scores = np.empty((n, kaudit), dtype=np.uint8)
        first_seed = 0 if smoke else cfg['first_seed']
        seeds = np.arange(first_seed, first_seed+n, dtype=np.int64)
        for si, seed in enumerate(seeds):
            # Separate development and audit streams; pair only within the run.
            drng = np.random.default_rng(np.random.SeedSequence([int(seed), fi, 101]))
            arng = np.random.default_rng(np.random.SeedSequence([int(seed), fi, 202]))
            train_p, train_y = make_tasks(drng, ktrain, g, family, cfg['corruptions'])
            tu = drng.uniform(size=(ktrain, bdev, 3))
            mu = drng.uniform(size=(checkpoints[-1], 3))
            ru = drng.uniform(size=(checkpoints[-1], 8))
            audit_p, audit_y = make_tasks(arng, kaudit, g, family, cfg['corruptions'])
            au = arng.uniform(size=(kaudit, budgets[-1], 3))
            initial_scores[si] = [64-(int(truth(p, inputs))^int(y)).bit_count() for p,y in zip(audit_p,audit_y)]
            seed_scores = repair_scores(SEED_POLICY, audit_p, audit_y, cards, tags, inputs, au, budgets)
            scores[si, 3] = np.broadcast_to(seed_scores, (len(checkpoints), kaudit, len(budgets)))
            for regime in range(3):
                snap, ds, acc = develop(regime, train_p, train_y, cards, tags, inputs, tu,
                                        mu, ru, checkpoints, bdev)
                policies[si, regime] = snap; dev_scores[si, regime] = ds
                acceptances[si, regime] = acc
                for ci, policy in enumerate(snap):
                    scores[si, regime, ci] = repair_scores(policy, audit_p, audit_y, cards,
                                                          tags, inputs, au, budgets)
            if (si+1) % 64 == 0 or si+1 == n:
                print(f'programs {family}: {si+1}/{n} independent runs', flush=True)
        elapsed = time.perf_counter()-started
        np.savez_compressed(out/f'programs_{family}_raw.npz', scores=scores, policies=policies,
                            dev_scores=dev_scores, acceptances=acceptances,
                            initial_scores=initial_scores, seeds=seeds,
                            checkpoints=checkpoints, budgets=budgets)
        means = scores.mean(axis=3)/64
        names = ['recursive', 'fixed_author', 'random_search', 'seed']
        for ci, c in enumerate(checkpoints):
            for bi, b in enumerate(budgets):
                for name, left, right in [('recursion_dividend', 0, 1), ('recursive_gain', 0, 3),
                                           ('fixed_gain', 1, 3), ('random_gain', 2, 3),
                                           ('recursive_minus_random', 0, 2)]:
                    z = means[:, left, ci, bi]-means[:, right, ci, bi]
                    row = {'family': family, 'C': int(c), 'B': int(b), 'contrast': name}
                    row.update(summarize(z, comparisons=27 if name=='recursion_dividend' else 135))
                    row['value_left'] = float(means[:, left, ci, bi].mean())
                    row['value_right'] = float(means[:, right, ci, bi].mean())
                    rows.append(row)
                for si, seed in enumerate(seeds):
                    raw_rows.append({'family': family, 'seed': int(seed), 'C': int(c), 'B': int(b),
                                     **{name: float(means[si, j, ci, bi]) for j, name in enumerate(names)}})
        resources.append({'family': family, 'independent_runs': n,
                          'wall_seconds': elapsed, 'gates': g,
                          'policy_evaluations_per_regime_run': int(checkpoints[-1])+1,
                          'development_circuit_evaluations_per_regime_run': (int(checkpoints[-1])+1)*ktrain*(bdev+1),
                          'audit_circuit_evaluations_per_selected_policy': kaudit*(int(budgets[-1])+1),
                          'selection_training_tasks': ktrain, 'audit_tasks': kaudit})
        write_csv(out/'programs_summary.csv', rows)
        write_csv(out/'programs_per_run.csv', raw_rows)
        write_csv(out/'resources.csv', resources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['all','exact','sensitivity','statistics','programs'], default='all')
    parser.add_argument('--protocol', type=Path, default=ROOT/'protocol.json')
    parser.add_argument('--out', type=Path, default=ROOT/'results')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    out = args.out
    if args.smoke and out.resolve() == (ROOT/'results').resolve():
        out = ROOT/'smoke_results'
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    selected = ['exact','sensitivity','statistics','programs'] if args.suite=='all' else [args.suite]
    for suite in selected:
        print(f'Running {suite} (smoke={args.smoke})', flush=True)
        if suite == 'sensitivity':
            run_sensitivity(out)
        elif suite == 'programs':
            run_programs(out, cfg[suite], args.smoke)
        else:
            globals()['run_'+suite](out, cfg[suite], args.smoke)
    import scipy, numba
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'suite': args.suite,
                'smoke': args.smoke, 'wall_seconds': time.perf_counter()-start,
                'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT/'src').glob('*.py'))},
                'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'numba': numba.__version__, 'platform': platform.platform(),
                'cpu_count': os.cpu_count(), 'gpu_used': False,
                'threading_note': 'Single-process CPU; no parallel=True or GPU kernels'}
    (out/f'metadata_{args.suite}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Finished {args.suite}; artifacts in {out}', flush=True)


if __name__ == '__main__':
    main()
