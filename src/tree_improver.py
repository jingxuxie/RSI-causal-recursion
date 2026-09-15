"""Typed conditional-program improvers; evaluator and interpreter are protected.

A complete depth-two decision tree has three (feature, threshold) nodes and four
leaves. Its ten integers are executable syntax, not the original weight table.
The same tree-driven typed edit acts on circuit genomes and tree syntax.
Regimes: 0 inherited author, 1 seed author, 2 uniform whole-program search.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from .boolean_core import active_genes, popcount, truth

# Features: time, stagnation, score, active fraction, normalized mean gene.
TREE_CARDS = np.array([5, 9, 5, 9, 5, 9, 24, 24, 24, 24], dtype=np.int64)
TREE_TAGS = np.array([0, 1, 0, 1, 0, 1, 2, 2, 2, 2], dtype=np.int64)
# Leaf 9 means any gene, uniform replacement, without an active-only restriction.
SEED_TREE = np.array([0, 2, 1, 4, 2, 4, 9, 9, 9, 9], dtype=np.int64)


def validate_tree(tree: np.ndarray) -> None:
    if tree.shape != (10,) or not np.issubdtype(tree.dtype, np.integer):
        raise ValueError('A tree must have exactly ten integer genes')
    if np.any(tree < 0) or np.any(tree >= TREE_CARDS):
        raise ValueError('Tree violates its typed grammar')


@njit(cache=True)
def tree_action(tree, features):
    node = 0
    for _ in range(2):
        right = features[tree[2 * node]] >= tree[2 * node + 1] / 8.0
        node = 2 * node + 1 + int(right)
    return tree[6 + node - 3]


@njit(cache=True)
def features_for(genome, cards, active, elapsed, stall, score):
    f = np.empty(5, dtype=np.float64)
    f[0] = elapsed
    f[1] = min(stall, 8) / 8.0
    f[2] = score
    f[3] = active.sum() / len(active)
    f[4] = 0.0
    for j in range(len(genome)):
        f[4] += genome[j] / max(cards[j] - 1, 1)
    f[4] /= len(genome)
    return f


@njit(cache=True)
def tree_edit(genome, cards, tags, active, author, features, uniforms):
    action = tree_action(author, features)
    only_active = action // 12
    tag = (action % 12) // 3
    operation = action % 3
    eligible = np.empty(len(genome), dtype=np.int64)
    n = 0
    for j in range(len(genome)):
        if (tag == 3 or tags[j] == tag) and (not only_active or active[j]):
            eligible[n] = j
            n += 1
    # A declared total-function fallback, identical at task and meta level.
    if n == 0:
        for j in range(len(genome)):
            eligible[j] = j
        n = len(genome)
    position = eligible[min(int(uniforms[0] * n), n - 1)]
    candidate = genome.copy()
    if operation == 0:
        candidate[position] = min(int(uniforms[1] * cards[position]), cards[position] - 1)
    elif operation == 1:
        candidate[position] = (candidate[position] + 1) % cards[position]
    else:
        candidate[position] = (candidate[position] - 1) % cards[position]
    return candidate


@njit(cache=True)
def tree_repair(tree, parents, targets, cards, tags, inputs, uniforms, budgets):
    """Each budget is a separate restart: time feature uses that exact budget.

    Thus a B=32 run is not a prefix of a B=64 run. Identical uniforms may still
    couple the two runs for variance reduction, without equating their policies.
    """
    out = np.empty((len(parents), len(budgets)), dtype=np.uint8)
    for k in range(len(parents)):
        for bi in range(len(budgets)):
            budget = budgets[bi]
            current = parents[k].copy()
            score = 64 - popcount(truth(current, inputs) ^ targets[k])
            stall = 0
            for t in range(budget):
                active = active_genes(current)
                f = features_for(current, cards, active, t / budget, stall, score / 64.0)
                candidate = tree_edit(current, cards, tags, active, tree, f, uniforms[k, t])
                new_score = 64 - popcount(truth(candidate, inputs) ^ targets[k])
                if new_score > score:
                    current = candidate
                    score = new_score
                    stall = 0
                else:
                    stall += 1
            out[k, bi] = score
    return out


@njit(cache=True)
def develop_tree(regime, parents, targets, cards, tags, inputs, task_uniforms,
                 meta_uniforms, random_uniforms, checkpoints, dev_budget):
    """Fixed development horizon checkpoints[-1]; checkpoints share one run.

    All proposals (including duplicate syntax and functional no-ops) are evaluated.
    Trees are retained only on strict training improvement. No final audit leaks.
    """
    current = SEED_TREE.copy()
    budget_array = np.array([dev_budget], dtype=np.int64)
    score = tree_repair(current, parents, targets, cards, tags, inputs,
                        task_uniforms, budget_array).sum()
    snapshots = np.empty((len(checkpoints), 10), dtype=np.int64)
    train_scores = np.empty(len(checkpoints), dtype=np.int64)
    accepted = np.empty(len(checkpoints), dtype=np.int64)
    diagnostics = np.zeros((len(checkpoints), 3), dtype=np.int64)
    active = np.ones(10, dtype=np.bool_)
    at = 0; n_accepted = 0; stall = 0
    changed_condition = 0; changed_leaf = 0; syntax_noop = 0
    for t in range(checkpoints[-1]):
        if regime == 2:
            candidate = np.empty_like(current)
            for j in range(10):
                candidate[j] = min(int(random_uniforms[t, j] * TREE_CARDS[j]), TREE_CARDS[j] - 1)
        else:
            author = SEED_TREE if regime == 1 else current
            f = features_for(current, TREE_CARDS, active, t / checkpoints[-1],
                             stall, score / (64.0 * len(parents)))
            candidate = tree_edit(current, TREE_CARDS, TREE_TAGS, active,
                                  author, f, meta_uniforms[t])
        changed_condition += int(np.any(candidate[:6] != current[:6]))
        changed_leaf += int(np.any(candidate[6:] != current[6:]))
        syntax_noop += int(np.all(candidate == current))
        candidate_score = tree_repair(candidate, parents, targets, cards, tags, inputs,
                                      task_uniforms, budget_array).sum()
        if candidate_score > score:
            current = candidate
            score = candidate_score
            n_accepted += 1
            stall = 0
        else:
            stall += 1
        if t + 1 == checkpoints[at]:
            snapshots[at] = current
            train_scores[at] = score
            accepted[at] = n_accepted
            diagnostics[at, 0] = changed_condition
            diagnostics[at, 1] = changed_leaf
            diagnostics[at, 2] = syntax_noop
            at += 1
            if at == len(checkpoints):
                break
    return snapshots, train_scores, accepted, diagnostics
