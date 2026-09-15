"""An independent, unpacked Boolean reference for complete task-level rollouts."""
import numpy as np
import pytest
from src import programs as upstream
from src.boolean_core import circuit_schema, input_tables, make_tasks
from src.tree_improver import TREE_CARDS, tree_repair


def scalar_score(circuit, target):
    correct = 0
    for bits in range(64):
        values = [bool((bits >> j) & 1) for j in range(6)]
        for j in range((len(circuit)-1)//3):
            op, a, b = map(int, circuit[3*j:3*j+3])
            x, y = values[a], values[b]
            values.append([x and y, x or y, x != y, not (x and y)][op])
        correct += values[int(circuit[-1])] == bool((int(target) >> bits) & 1)
    return correct


def scalar_active(circuit):
    answer = np.zeros(len(circuit), dtype=bool)
    answer[-1] = True
    stack = [int(circuit[-1])]
    while stack:
        node = stack.pop()
        if node < 6:
            continue
        j = 3*(node-6)
        if answer[j]:
            continue
        answer[j:j+3] = True
        stack.extend(map(int, circuit[j+1:j+3]))
    return answer


def reference_repair(tree, parent, target, cards, tags, uniforms, budget):
    current = parent.copy()
    score = scalar_score(current, target)
    stall = 0
    for step in range(budget):
        active = scalar_active(current)
        features = [step/budget, min(stall, 8)/8, score/64,
                    sum(active)/len(active),
                    sum(float(g)/max(int(c)-1, 1) for g, c in zip(current, cards))/len(current)]
        right = features[int(tree[0])] >= tree[1]/8
        offset = 4 if right else 2
        child_right = features[int(tree[offset])] >= tree[offset+1]/8
        action = int(tree[6+2*int(right)+int(child_right)])
        only_active, remainder = divmod(action, 12)
        tag, operation = divmod(remainder, 3)
        positions = [j for j in range(len(current))
                     if (tag == 3 or tags[j] == tag) and (not only_active or active[j])]
        positions = positions or list(range(len(current)))
        j = positions[min(int(uniforms[step, 0]*len(positions)), len(positions)-1)]
        candidate = current.copy()
        if operation == 0:
            candidate[j] = min(int(uniforms[step, 1]*cards[j]), cards[j]-1)
        else:
            candidate[j] = (candidate[j] + (1 if operation == 1 else -1)) % cards[j]
        candidate_score = scalar_score(candidate, target)
        if candidate_score > score:
            current, score, stall = candidate, candidate_score, 0
        else:
            stall += 1
    return score


@pytest.mark.parametrize('seed', range(5))
def test_whole_rollout_against_independent_scalar_reference(seed):
    rng = np.random.default_rng(5550+seed)
    parents, targets = make_tasks(rng, 2, 6, 'mixed')
    cards, tags = circuit_schema(6)
    uniforms = rng.random((2, 12, 2))
    for _ in range(4):
        tree = np.array([rng.integers(c) for c in TREE_CARDS])
        actual = tree_repair(tree, parents, targets, cards, tags,
                             input_tables(), uniforms, np.array([5, 12]))
        expected = [[reference_repair(tree, p, y, cards, tags, u, b)
                     for b in [5, 12]] for p, y, u in zip(parents, targets, uniforms)]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('family', ['mixed', 'xor', 'logic'])
def test_extracted_core_matches_unchanged_original_task_generator(family):
    a = make_tasks(np.random.default_rng(7654), 24, 16, family)
    b = upstream.make_tasks(np.random.default_rng(7654), 24, 16, family)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
