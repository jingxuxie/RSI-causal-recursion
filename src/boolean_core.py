"""Boolean repair task generator shared by the new tree experiments.

Extracted from the declared six-input circuit semantics; no model/API access.
"""
from __future__ import annotations
import numpy as np
from numba import njit

def input_tables() -> np.ndarray:
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


def make_tasks(rng: np.random.Generator, count: int, gates: int,
               family: str, corruptions: int = 5) -> tuple[np.ndarray, np.ndarray]:
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


