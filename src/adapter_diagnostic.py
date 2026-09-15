"""A declared intervention on meta-level gene types; task semantics stay fixed."""
from __future__ import annotations
import numpy as np
from numba import njit
from .tree_improver import (TREE_CARDS, TREE_TAGS, SEED_TREE, tree_repair,
                           tree_edit, features_for)


@njit(cache=True)
def develop_adapter(recursive, meta_tags, parents, targets, cards, tags, inputs,
                    task_uniforms, meta_uniforms, dev_budget):
    current = SEED_TREE.copy()
    budget = np.array([dev_budget], dtype=np.int64)
    score = tree_repair(current, parents, targets, cards, tags, inputs,
                        task_uniforms, budget).sum()
    active = np.ones(10, dtype=np.bool_)
    stall = 0; accepted = 0
    for t in range(len(meta_uniforms)):
        author = current if recursive else SEED_TREE
        f = features_for(current, TREE_CARDS, active, t / len(meta_uniforms),
                         stall, score / (64.0 * len(parents)))
        candidate = tree_edit(current, TREE_CARDS, meta_tags, active, author,
                              f, meta_uniforms[t])
        candidate_score = tree_repair(candidate, parents, targets, cards, tags,
                                      inputs, task_uniforms, budget).sum()
        if candidate_score > score:
            current = candidate
            score = candidate_score
            accepted += 1
            stall = 0
        else:
            stall += 1
    return current, score, accepted
