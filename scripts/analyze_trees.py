#!/usr/bin/env python3
"""Prespecified finite-sample primary intervals; retain all secondary grid cells."""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.shared_adapter import consistency_example, persistent_interval, rectangular_interval


def empirical_bernstein(x: np.ndarray, delta: float, comparisons: int) -> tuple[float, float, float]:
    """Two-sided Maurer--Pontil radius for [-1,1], Bonferroni over comparisons.

    The mean of descendants from one development run is ONE bounded observation.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 or np.any(np.abs(x) > 1) or not 0 < delta < 1 or comparisons < 1:
        raise ValueError('Invalid bounded-data confidence request')
    log = math.log(4 * comparisons / delta)
    radius = math.sqrt(2 * x.var(ddof=1) * log / n) + 14 * log / (3 * (n - 1))
    mean = float(x.mean())
    return mean, max(-1., mean-radius), min(1., mean+radius)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results/trees')
    args = p.parse_args(); out = args.results
    meta = json.loads((out / 'metadata.json').read_text()); cfg = meta['executed_config']
    raw = np.load(out / 'raw.npz', allow_pickle=False)
    score = raw['scores'].astype(float) / meta['score_denominator']
    n = score.shape[1]
    ci = list(raw['cs']).index(cfg['primary_C']); bi = list(raw['bs']).index(cfg['primary_B'])
    summary = {'primary': [], 'seed_gains': [], 'secondary': [], 'diagnostics': [],
               'n_per_family': n, 'comparison_family_error': cfg['confidence_delta']}
    for fi, family in enumerate(cfg['families']):
        v = score[fi, :, :, ci, bi]
        mean, lo, hi = empirical_bernstein(v[:, 0]-v[:, 1], cfg['confidence_delta'], cfg['primary_contrasts'])
        summary['primary'].append({'family': family, 'mean': mean, 'lower': lo, 'upper': hi,
                                   'mean_R': float(v[:, 0].mean()), 'mean_F': float(v[:, 1].mean()),
                                   'mean_random': float(v[:, 2].mean()), 'mean_seed': float(v[:, 3].mean())})
        for regime, name in enumerate(cfg['regimes']):
            mean, lo, hi = empirical_bernstein(v[:, regime]-v[:, 3], cfg['confidence_delta'], cfg['seed_gain_contrasts'])
            summary['seed_gains'].append({'family': family, 'regime': name, 'mean': mean, 'lower': lo, 'upper': hi})
            d = raw['diagnostics'][fi, :, regime, ci].mean(axis=0)
            summary['diagnostics'].append({'family': family, 'regime': name,
                'condition_edits': float(d[0]), 'leaf_edits': float(d[1]), 'syntax_noops': float(d[2]),
                'accepted': float(raw['accepted'][fi, :, regime, ci].mean()),
                'training_score': float(raw['train'][fi, :, regime, ci].mean()/(64*cfg['training_tasks']))})
        for cj, c in enumerate(raw['cs']):
            for bj, b in enumerate(raw['bs']):
                delta = score[fi, :, 0, cj, bj] - score[fi, :, 1, cj, bj]
                summary['secondary'].append({'family': family, 'C_checkpoint': int(c), 'B': int(b),
                    'mean': float(delta.mean()), 'descriptive_standard_error': float(delta.std(ddof=1)/math.sqrt(n))})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'primary_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary['primary'][0])); w.writeheader(); w.writerows(summary['primary'])
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Family & $D_R$ & $D_F$ & $\Gamma$ [simultaneous 95\% interval] \\', r'\midrule']
    for r in summary['primary']:
        dr = 100*(r['mean_R']-r['mean_seed']); df = 100*(r['mean_F']-r['mean_seed'])
        lines.append(f"{r['family']} & {dr:.2f} & {df:.2f} & {100*r['mean']:.2f} [{100*r['lower']:.2f}, {100*r['upper']:.2f}] " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table_name = 'trees_table.tex' if out.name == 'trees_confirmation' else 'trees_exploration_table.tex'
    (ROOT / 'paper' / table_name).write_text('\n'.join(lines)+'\n')
    model = consistency_example(.2)
    theory = {'persistent_interval': persistent_interval(*model),
              'rowwise_relaxation': rectangular_interval(*model)}
    (out / 'persistent_adapter.json').write_text(json.dumps(theory, indent=2)+'\n')
    print(json.dumps({'primary': summary['primary'], 'seed_gains': summary['seed_gains'],
                      'persistent_adapter': theory}, indent=2))


if __name__ == '__main__':
    main()
