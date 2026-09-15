#!/usr/bin/env python3
"""Retain all adapter interventions and account for both confidence families."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_trees import empirical_bernstein


def main():
    out=ROOT/'results/adapter_diagnostic'
    meta=json.loads((out/'metadata.json').read_text());cfg=meta['executed_config']
    raw=np.load(out/'raw.npz',allow_pickle=False)
    integer=raw['scores'].astype(np.int64)
    v=integer/meta['denominator']
    rows=[]
    for fi,family in enumerate(cfg['families']):
        for j,permutation in enumerate(cfg['adapter_permutations']):
            gd=integer[fi,:,j]-integer[fi,:,-1]
            md,ld,ud=empirical_bernstein(gd/meta['denominator'],cfg['confidence_delta'],cfg['dividend_comparisons'])
            ae=integer[fi,:,j]-integer[fi,:,0]
            ma,la,ua=(0.,0.,0.) if j==0 else empirical_bernstein(
                ae/meta['denominator'],cfg['confidence_delta'],cfg['adapter_effect_comparisons'])
            rows.append({'family':family,'permutation':permutation,'dividend':md,
                'dividend_lower':ld,'dividend_upper':ud,'adapter_effect':ma,
                'adapter_lower':la,'adapter_upper':ua,
                'mean_recursive':float(v[fi,:,j].mean()),
                'mean_fixed':float(v[fi,:,-1].mean()),
                'sum_dividend_integer':int(gd.sum()),'sum_squared_dividend_integer':int((gd*gd).sum()),
                'sum_adapter_integer':int(ae.sum()),'sum_squared_adapter_integer':int((ae*ae).sum())})
    summary={'n':v.shape[1],'denominator':meta['denominator'],'rows':rows,
       'note':'15 nonidentity adapter effects and 18 dividends have separate simultaneous 95% families. Identity adapter effect is deterministically zero.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=[r'\begin{tabular}{llrr}',r'\toprule',
           r'Family & Mapping & $\Gamma$ [95\% interval] & Change from identity [95\% interval] \\',r'\midrule']
    for r in rows:
        perm=''.join(map(str,r['permutation']))
        gd=f"{100*r['dividend']:.2f} [{100*r['dividend_lower']:.2f}, {100*r['dividend_upper']:.2f}]"
        ga='0 (identity)' if perm=='012' else f"{100*r['adapter_effect']:.2f} [{100*r['adapter_lower']:.2f}, {100*r['adapter_upper']:.2f}]"
        lines.append(f"{r['family']} & {perm} & {gd} & {ga} "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/adapter_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
