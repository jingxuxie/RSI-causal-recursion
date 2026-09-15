#!/usr/bin/env python3
"""Check all interface interventions; keep evidence verification distinct from review."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_trees import interval_from_sufficient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args();out=ROOT/'results/adapter_diagnostic'
    meta=json.loads((out/'metadata.json').read_text())
    summary=json.loads((out/'summary.json').read_text());cfg=meta['executed_config']
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(ROOT/'protocol_adapter_diagnostic.json')==meta['protocol_sha256']
    n,den=summary['n'],summary['denominator']
    assert n==cfg['development_runs_per_family'] and len(summary['rows'])==18
    checks=[]
    for row in summary['rows']:
        for kind,j,fields in [('dividend',cfg['dividend_comparisons'],['dividend','dividend_lower','dividend_upper']),
                              ('adapter',cfg['adapter_effect_comparisons'],['adapter_effect','adapter_lower','adapter_upper'])]:
            stats={'sum_integer':row[f'sum_{kind}_integer'],
                   'sum_squared_integer':row[f'sum_squared_{kind}_integer']}
            if kind=='adapter' and row['permutation']==[0,1,2]:
                assert stats=={'sum_integer':0,'sum_squared_integer':0}
                expected=(0.,0.,0.)
            else:
                expected=interval_from_sufficient(stats,n,den,j,cfg['confidence_delta'])
            np.testing.assert_allclose(expected,[row[f] for f in fields],atol=1e-14)
    checks.append('All 18 dividends and 15 nonidentity adapter intervals recompute from exact integer statistics')
    if not args.aggregate_only:
        assert sha(out/'raw.npz')==meta['raw_sha256']
        for path,h in meta['source_sha256'].items():
            assert sha(ROOT/path)==h,path
        with np.load(out/'raw.npz',allow_pickle=False) as raw:
            v=raw['scores'].astype(np.int64)
            for row in summary['rows']:
                fi=cfg['families'].index(row['family']);j=cfg['adapter_permutations'].index(row['permutation'])
                for kind,z in [('dividend',v[fi,:,j]-v[fi,:,-1]),('adapter',v[fi,:,j]-v[fi,:,0])]:
                    assert int(z.sum())==row[f'sum_{kind}_integer']
                    assert int((z*z).sum())==row[f'sum_squared_{kind}_integer']
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([sys.executable,str(ROOT/'scripts/run_adapter_diagnostic.py'),
                   '--runs','2','--output',tmp],check=True,capture_output=True)
                with np.load(Path(tmp)/'raw.npz',allow_pickle=False) as replay:
                    for name in ['scores','policies','training','accepted']:
                        np.testing.assert_array_equal(raw[name][:,:2],replay[name])
        checks.append('Source and raw hashes verified; complete outcomes for two seeds in every family replay bitwise')
    report={'status':'PASS','aggregate_only':args.aggregate_only,'checks':checks}
    target='aggregate_verification.json' if args.aggregate_only else 'verification.json'
    (out/target).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
