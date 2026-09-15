#!/usr/bin/env python3
"""Build paper tables from saved raw outcomes; never generates replacement data."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.theory import empirical_bernstein_interval, shared_history_interval


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT/'results/confirmation'
    protocol = json.loads((ROOT/'protocol_confirmation.json').read_text())
    expected_n = protocol['programs']['development_runs']
    records=[]
    for family in protocol['programs']['families']:
        data=np.load(out/f'programs_{family}_raw.npz')
        assert len(data['seeds'])==expected_n
        assert np.array_equal(data['checkpoints'],[128]) and np.array_equal(data['budgets'],[32])
        values=data['scores'].mean(axis=3)[:, :, 0, 0]/64
        for name,left,right,J in [('recursion_dividend',0,1,3),('recursive_gain',0,3,9),
                                   ('fixed_gain',1,3,9),('random_gain',2,3,9),
                                   ('recursive_minus_random',0,2,3)]:
            mean,lo,hi=empirical_bernstein_interval(values[:,left]-values[:,right], comparisons=J)
            records.append(dict(family=family,contrast=name,n=expected_n,mean=mean,
                                eb_lower=lo,eb_upper=hi,simultaneous_comparisons=J,
                                alpha=.05,value_left=values[:,left].mean(),value_right=values[:,right].mean(),
                                equivalence_1pp=(lo>-.01 and hi<.01) if name=='recursion_dividend' else '',
                                role='posthoc_descriptive_bound' if name=='recursive_minus_random' else 'prespecified'))
    with open(out/'certificates.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    def pick(f,c): return next(r for r in records if r['family']==f and r['contrast']==c)
    tex=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Family & $D_R$ & $D_F$ & $\Gamma$ & 95\% interval for $\Gamma$ \\',r'\midrule']
    for fam in protocol['programs']['families']:
        r=pick(fam,'recursion_dividend')
        tex.append(f"{fam.capitalize()} & {100*pick(fam,'recursive_gain')['mean']:.2f} & {100*pick(fam,'fixed_gain')['mean']:.2f} & {100*r['mean']:+.3f} & $[{100*r['eb_lower']:+.3f},{100*r['eb_upper']:+.3f}]$ \\")
    # Append the second TeX line-break slash without fragile escaping.
    tex=[t+'\\' if t.endswith(' \\') and not t.endswith(' \\\\') else t for t in tex]
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/confirmation_table.tex').write_text('\n'.join(tex)+'\n')
    gains=[r'\begin{tabular}{llrr}',r'\toprule',r'Family & Method versus seed & Gain (pp) & Simultaneous 95\% interval (pp) \\',r'\midrule']
    for fam in protocol['programs']['families']:
        for name,label in [('recursive_gain','Recursive'),('fixed_gain','Fixed author'),('random_gain','Random search')]:
            r=pick(fam,name)
            gains.append(f"{fam.capitalize()} & {label} & {100*r['mean']:.2f} & $[{100*r['eb_lower']:.2f},{100*r['eb_upper']:.2f}]$"+r' \\')
    gains.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/gains_table.tex').write_text('\n'.join(gains)+'\n')
    sensitivity=rows(ROOT/'results/sensitivity.csv')
    r=next(r for r in sensitivity if r['location']=='rare' and abs(float(r['epsilon'])-.1)<1e-8)
    lineage=rows(ROOT/'results/lineage.csv')[-1]
    macros={'NominalContrast':f"${float(r['nominal']):.4f}$",
            'RareInterval':f"$[{float(r['lower']):.4f},{float(r['upper']):.4f}]$",
            'CoarseInterval':f"$[{float(r['uniform_lower']):.4f},{float(r['uniform_upper']):.4f}]$",
            'NaiveCoverage':f"{100*float(lineage['naive_hoeffding_coverage']):.2f}"}
    (ROOT/'paper/results_macros.tex').write_text('\n'.join(chr(92)+'newcommand{'+chr(92)+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    shared=shared_history_interval([.5,.5],[.3,.9],[.2,.8],1.)
    (ROOT/'results/shared_history.json').write_text(json.dumps({'common_history_interval':shared,
        'independent_history_relaxation':[-.5,.7],'r_values':[.3,.9],'f_values':[.2,.8],
        'nominal_history':[.5,.5],'radius':1.0},indent=2)+'\n')
    print(json.dumps({'all_three_equivalent':all(pick(f,'recursion_dividend')['equivalence_1pp'] for f in protocol['programs']['families']),
                      'certificates':records},indent=2))

if __name__=='__main__': main()
