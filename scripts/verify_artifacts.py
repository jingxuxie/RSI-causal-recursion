#!/usr/bin/env python3
"""Verify saved evidence and replay independent seed prefixes without overwriting it."""
from pathlib import Path
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import tempfile
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/run_experiments.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    checks=[]
    for protocol_path,source in [(ROOT/'protocol.json',ROOT/'results'),
                                  (ROOT/'protocol_confirmation.json',ROOT/'results/confirmation')]:
        p=json.loads(protocol_path.read_text())
        meta_name='metadata_all.json' if source.name=='results' else 'metadata_programs.json'
        meta=json.loads((source/meta_name).read_text())
        assert meta['protocol_sha256']==sha(protocol_path)
        assert meta['source_sha256']['src/programs.py']==sha(ROOT/'src/programs.py')
        checks.append(f'{protocol_path.name}: protocol and executable-program hashes match')
        cfg=dict(p['programs']);cfg['development_runs']=2
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            runner.run_programs(Path(tmp),cfg,False)
            for family in cfg['families']:
                stored=np.load(source/f'programs_{family}_raw.npz')
                replay=np.load(Path(tmp)/f'programs_{family}_raw.npz')
                for key in ['scores','policies','dev_scores','acceptances','initial_scores','seeds']:
                    assert np.array_equal(stored[key][:2],replay[key]), (family,key)
                checks.append(f'{protocol_path.name}/{family}: first two seed sets replay bitwise exactly')
    # All confidence values must be recomputable from the actual saved integer outcomes.
    from src.theory import empirical_bernstein_interval
    with open(ROOT/'results/confirmation/certificates.csv',newline='') as f:
        certificates=list(csv.DictReader(f))
    mapping={'recursion_dividend':(0,1),'recursive_gain':(0,3),'fixed_gain':(1,3),
             'random_gain':(2,3),'recursive_minus_random':(0,2)}
    for row in certificates:
        data=np.load(ROOT/'results/confirmation'/f"programs_{row['family']}_raw.npz")
        v=data['scores'].mean(axis=3)[:,:,0,0]/64
        left,right=mapping[row['contrast']]
        interval=empirical_bernstein_interval(v[:,left]-v[:,right],comparisons=int(row['simultaneous_comparisons']))
        assert np.allclose(interval,[float(row[k]) for k in ['mean','eb_lower','eb_upper']],atol=1e-14)
    checks.append('All 15 confirmation summaries and bounds recompute from saved integer scores')
    raw_files=sorted((ROOT/'results').rglob('*_raw.npz'))
    manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in raw_files}
    report={'status':'PASS','checks':checks,'raw_data':manifest,
            'current_source':{str(p.relative_to(ROOT)):sha(p) for folder in ['src','scripts','tests'] for p in sorted((ROOT/folder).glob('*.py'))},
            'note':'Checks validate implementation and recorded evidence, not novelty or independent proof review.'}
    (ROOT/'results/verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
