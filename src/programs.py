"""Bounded self-referential mutation policies on executable Boolean circuits.

There is no eval/exec, network access, language model, or writable evaluator.
A fixed interpreter reads an eight-integer policy. The same edit primitive
mutates a circuit or the policy itself. R inherits the selected policy as author;
F edits the same incumbent using the original author. Both can improve policies.
"""
from __future__ import annotations
import numpy as np
from numba import njit

SEED_POLICY = np.array([2, 2, 2, 2, 2, 2, 0, 0], dtype=np.int64)
POLICY_CARDS = np.array([5, 5, 5, 5, 5, 5, 4, 2], dtype=np.int64)
POLICY_TAGS = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)


def input_tables() -> np.ndarray:
    """All 64 inputs for six Boolean variables, packed into uint64 words."""
    return np.array([sum(((x >> j) & 1) << x for x in range(64))
                     for j in range(6)], dtype=np.uint64)


@njit(cache=True)
def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def truth(circuit, inputs):
    g = (len(circuit) - 1) // 3
    values = np.empty(6 + g, dtype=np.uint64)
    values[:6] = inputs
    for j in range(g):
        a = values[circuit[3*j + 1]]; b = values[circuit[3*j + 2]]
        op = circuit[3*j]
        if op == 0:
            values[6+j] = a & b
        elif op == 1:
            values[6+j] = a | b
        elif op == 2:
            values[6+j] = a ^ b
        else:
            values[6+j] = ~(a & b)
    return values[circuit[-1]]


@njit(cache=True)
def active_genes(circuit):
    g = (len(circuit) - 1) // 3
    nodes = np.zeros(6 + g, dtype=np.bool_)
    nodes[circuit[-1]] = True
    active = np.zeros(len(circuit), dtype=np.bool_)
    active[-1] = True
    for j in range(g - 1, -1, -1):
        if nodes[6+j]:
            active[3*j:3*j+3] = True
            nodes[circuit[3*j+1]] = True
            nodes[circuit[3*j+2]] = True
    return active


@njit(cache=True)
def mutate(genome, cards, tags, active, author, uniforms):
    """One bounded edit; every branch uses the same protected instruction set."""
    weights = np.empty(len(genome), dtype=np.float64)
    for j in range(len(genome)):
        weights[j] = (1 << author[tags[j]]) * ((1 << author[6]) if active[j] else 1)
    threshold = uniforms[0] * weights.sum()
    position = len(genome) - 1
    cumulative = 0.
    for j in range(len(genome)):
        cumulative += weights[j]
        if threshold < cumulative:
            position = j
            break
    mode_weights = np.array([1 << author[3], 1 << author[4], 1 << author[5]])
    threshold = uniforms[1] * mode_weights.sum()
    mode = 2
    if threshold < mode_weights[0]:
        mode = 0
    elif threshold < mode_weights[0] + mode_weights[1]:
        mode = 1
    result = genome.copy()
    if mode == 0:
        result[position] = min(int(uniforms[2] * cards[position]), cards[position] - 1)
    elif mode == 1:
        result[position] = (result[position] + 1) % cards[position]
    else:
        result[position] = (result[position] - 1) % cards[position]
    return result


def circuit_schema(gates: int) -> tuple[np.ndarray, np.ndarray]:
    if gates < 1:
        raise ValueError('Positive gate count required')
    cards = np.empty(3*gates + 1, dtype=np.int64)
    tags = np.empty_like(cards)
    for j in range(gates):
        cards[3*j:3*j+3] = [4, 6+j, 6+j]
        tags[3*j:3*j+3] = [0, 1, 1]
    cards[-1] = 6 + gates; tags[-1] = 2
    return cards, tags


@njit(cache=True)
def repair_scores(policy, parents, targets, cards, tags, inputs, uniforms, budgets):
    """Scores of the ACTUALLY selected circuit, not an oracle-selected descendant.

    All 64 inputs are visible to this within-task exact evaluator. Generalization
    is across fresh tasks and parents, not a hidden split within each truth table.
    Neutral moves depend on policy[7]; the protected score is never modified.
    """
    out = np.empty((len(parents), len(budgets)), dtype=np.uint8)
    for k in range(len(parents)):
        current = parents[k].copy()
        score = 64 - popcount(truth(current, inputs) ^ targets[k])
        at = 0
        for b in range(1, budgets[-1] + 1):
            active = active_genes(current)
            candidate = mutate(current, cards, tags, active, policy, uniforms[k, b-1])
            candidate_score = 64 - popcount(truth(candidate, inputs) ^ targets[k])
            if candidate_score > score or (candidate_score == score and policy[7] == 1):
                current = candidate
                score = candidate_score
            if b == budgets[at]:
                out[k, at] = score
                at += 1
                if at == len(budgets):
                    break
    return out


@njit(cache=True)
def develop(regime, parents, targets, cards, tags, inputs, task_uniforms,
            meta_uniforms, random_uniforms, checkpoints, dev_budget):
    """regime: 0=recursive, 1=fixed-author, 2=independent random policy search,
    3=fixed-interpreter emulation of recursive author inheritance.

    Training tasks AND search random numbers are fixed within a development run.
    Every proposed policy is evaluated, including no-op proposals; no caching.
    """
    current = SEED_POLICY.copy()
    budget_array = np.array([dev_budget], dtype=np.int64)
    score = repair_scores(current, parents, targets, cards, tags, inputs,
                          task_uniforms, budget_array).sum()
    snapshots = np.empty((len(checkpoints), len(current)), dtype=np.int64)
    scores = np.empty(len(checkpoints), dtype=np.int64)
    accepted = np.zeros(len(checkpoints), dtype=np.int64)
    active = np.ones(len(current), dtype=np.bool_)
    index = 0; n_accepted = 0
    for c in range(1, checkpoints[-1] + 1):
        if regime == 2:
            candidate = np.empty_like(current)
            for j in range(len(current)):
                candidate[j] = min(int(random_uniforms[c-1, j] * POLICY_CARDS[j]), POLICY_CARDS[j]-1)
        else:
            author = SEED_POLICY if regime == 1 else current
            candidate = mutate(current, POLICY_CARDS, POLICY_TAGS, active,
                               author, meta_uniforms[c-1])
        candidate_score = repair_scores(candidate, parents, targets, cards, tags,
                                        inputs, task_uniforms, budget_array).sum()
        if candidate_score > score:
            current = candidate
            score = candidate_score
            n_accepted += 1
        if c == checkpoints[index]:
            snapshots[index] = current
            scores[index] = score
            accepted[index] = n_accepted
            index += 1
            if index == len(checkpoints):
                break
    return snapshots, scores, accepted


def make_tasks(rng: np.random.Generator, count: int, gates: int,
               family: str, corruptions: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Generate targets plus corrupted parents; reject functionally perfect parents.

    Families change target-circuit operation frequencies, not the editing API.
    Targets that compute constants are rejected to avoid trivial repair tasks.
    """
    distributions = {'mixed': [.25, .25, .25, .25],
                     'xor': [.1, .1, .7, .1],
                     'logic': [.35, .35, .05, .25]}
    if family not in distributions or count < 1 or corruptions < 1:
        raise ValueError('Invalid task configuration')
    cards, _ = circuit_schema(gates)
    inputs = input_tables()
    parents = np.empty((count, len(cards)), dtype=np.int64)
    targets = np.empty(count, dtype=np.uint64)
    made = 0
    for _ in range(count * 10000):
        target = np.array([rng.integers(n) for n in cards], dtype=np.int64)
        target[0:-1:3] = rng.choice(4, size=gates, p=distributions[family])
        target[-1] = 6 + gates - 1
        goal = np.uint64(truth(target, inputs))
        ones = popcount(goal)
        if ones < 4 or ones > 60:
            continue
        parent = target.copy()
        # Corrupt active genes so the observed task is not usually a no-op.
        active = np.flatnonzero(active_genes(parent))
        for _j in range(corruptions):
            j = int(rng.choice(active))
            parent[j] = int(rng.integers(cards[j]))
        if truth(parent, inputs) == goal:
            continue
        parents[made] = parent; targets[made] = goal
        made += 1
        if made == count:
            return parents, targets
    raise RuntimeError('Task rejection sampler exhausted its explicit limit')
