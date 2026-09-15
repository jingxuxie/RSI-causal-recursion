#!/usr/bin/env python3
"""Run every declared meta-type permutation; never select just the favorable one."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.boolean_core import circuit_schema, input_tables, make_tasks
from src.tree_improver import TREE_TAGS, tree_repair
from src.adapter_diagnostic import develop_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'protocol_adapter_diagnostic.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/adapter_diagnostic')
    parser.add_argument('--runs', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text()); out = args.output
    n = args.runs or cfg['development_runs_per_family']
    if n < 2:
        raise ValueError('At least two runs are required')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'raw.npz').exists():
        raise FileExistsError('Refusing to overwrite existing outcomes')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources = {p:digest(ROOT/p) for p in ['src/boolean_core.py','src/tree_improver.py',
                  'src/adapter_diagnostic.py','scripts/run_adapter_diagnostic.py']}
    protocol_digest = digest(args.protocol)
    permutations = np.array(cfg['adapter_permutations'], dtype=np.int64)
    scores = np.zeros((len(cfg['families']), n, len(permutations)+1), dtype=np.uint16)
    policies = np.zeros((*scores.shape, 10), dtype=np.uint8)
    training = np.zeros_like(scores); accepted = np.zeros_like(scores)
    cards,tags = circuit_schema(cfg['gates']); inputs=input_tables()
    tr, au, bd = cfg['training_tasks'],cfg['audit_tasks'],cfg['training_budget']
    c,b = cfg['C'],cfg['B']; budget=np.array([b],dtype=np.int64)
    start=time.perf_counter()
    for fi,family in enumerate(cfg['families']):
        for i in range(n):
            streams=np.random.SeedSequence([cfg['master_seed'],fi,i]).spawn(5)
            rt,ra,ru,rm,rv=[np.random.default_rng(s) for s in streams]
            p,y=make_tasks(rt,tr,cfg['gates'],family)
            ap,ay=make_tasks(ra,au,cfg['gates'],family)
            ut,um,uv=ru.random((tr,bd,2)),rm.random((c,2)),rv.random((au,b,2))
            for j in range(len(permutations)+1):
                recursive=j<len(permutations)
                mt=permutations[j][TREE_TAGS] if recursive else TREE_TAGS
                tree,s,a=develop_adapter(recursive,mt,p,y,cards,tags,inputs,ut,um,bd)
                policies[fi,i,j]=tree;training[fi,i,j]=s;accepted[fi,i,j]=a
                scores[fi,i,j]=tree_repair(tree,ap,ay,cards,tags,inputs,uv,budget).sum()
            if (i+1)%128==0 or i+1==n:
                print(f'{family}: {i+1}/{n}, {time.perf_counter()-start:.1f}s',flush=True)
    np.savez_compressed(out/'raw.npz',scores=scores,policies=policies,
                        training=training,accepted=accepted,permutations=permutations)
    metadata={'executed_config':{**cfg,'development_runs_per_family':n},
       'protocol_sha256':protocol_digest,'source_sha256':sources,
       'source_unchanged':all(digest(ROOT/p)==v for p,v in sources.items()),
       'elapsed_seconds':time.perf_counter()-start,'denominator':au*64,
       'raw_sha256':digest(out/'raw.npz'),'python':sys.version,'numpy':np.__version__,
       'circuit_evaluations_per_development':(c+1)*tr*(bd+1),
       'circuit_evaluations_per_audit':au*(b+1),
       'note':'Additional fixed-author adapters are mathematically pointwise identical, tested independently, and are not counted as extra independent replications.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'complete':True,'elapsed':metadata['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    main()
