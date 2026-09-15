"""Independent interpreter and intervention checks for conditional programs."""
import numpy as np
import pytest
from src.boolean_core import circuit_schema, input_tables, make_tasks, truth
from src.tree_improver import (TREE_CARDS, TREE_TAGS, SEED_TREE, validate_tree,
                              tree_action, tree_edit, features_for, tree_repair,
                              develop_tree)


def reference_action(tree, features):
    root_right = features[int(tree[0])] >= tree[1] / 8
    offset = 4 if root_right else 2
    child_right = features[int(tree[offset])] >= tree[offset+1] / 8
    return int(tree[6 + 2 * int(root_right) + int(child_right)])


@pytest.mark.parametrize('seed', range(8))
def test_scalar_interpreter(seed):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        t = np.array([rng.integers(c) for c in TREE_CARDS])
        f = rng.random(5)
        assert tree_action(t, f) == reference_action(t, f)


def test_threshold_equality_goes_right():
    t = SEED_TREE.copy(); t[6:] = np.arange(4)
    assert tree_action(t, np.full(5, .5)) == 3


def test_validation():
    validate_tree(SEED_TREE)
    for x in [np.zeros(9, dtype=int), np.zeros(10), -np.ones(10, dtype=int), TREE_CARDS]:
        with pytest.raises(ValueError):
            validate_tree(x)


def test_typed_edit_and_no_input_mutation():
    rng = np.random.default_rng(11)
    g = SEED_TREE.copy(); original = g.copy()
    active = np.ones(10, dtype=np.bool_)
    for _ in range(1000):
        author = np.array([rng.integers(c) for c in TREE_CARDS])
        candidate = tree_edit(g, TREE_CARDS, TREE_TAGS, active, author, rng.random(5), rng.random(2))
        validate_tree(candidate)
        assert np.count_nonzero(candidate != g) <= 1
        assert np.array_equal(g, original)


def test_seed_uniform_edit_semantics():
    g = SEED_TREE.copy()
    for j in range(10):
        out = tree_edit(g, TREE_CARDS, TREE_TAGS, np.ones(10, dtype=bool),
                        SEED_TREE, np.zeros(5), np.array([(j+.2)/10, .01]))
        expected = g.copy(); expected[j] = 0
        assert np.array_equal(out, expected)


def fixture(seed=33):
    rng = np.random.default_rng(seed)
    parents, targets = make_tasks(rng, 3, 6, 'mixed')
    cards, tags = circuit_schema(6)
    return rng, parents, targets, cards, tags, input_tables()


def test_first_generation_identical_and_immutable():
    rng, p, y, c, tag, inp = fixture()
    oldp, oldy = p.copy(), y.copy()
    ut, um, ur = rng.random((3, 8, 2)), rng.random((1, 2)), rng.random((1, 10))
    a = develop_tree(0, p, y, c, tag, inp, ut, um, ur, np.array([1]), 8)
    b = develop_tree(1, p, y, c, tag, inp, ut, um, ur, np.array([1]), 8)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)
    np.testing.assert_array_equal(p, oldp); np.testing.assert_array_equal(y, oldy)


def test_training_selection_monotone_and_replayable():
    rng, p, y, c, tag, inp = fixture(55)
    ut, um, ur = rng.random((3, 8, 2)), rng.random((16, 2)), rng.random((16, 10))
    checkpoints = np.array([1, 4, 8, 16])
    for regime in range(3):
        a = develop_tree(regime, p, y, c, tag, inp, ut, um, ur, checkpoints, 8)
        b = develop_tree(regime, p, y, c, tag, inp, ut, um, ur, checkpoints, 8)
        for x, z in zip(a, b):
            np.testing.assert_array_equal(x, z)
        assert np.all(np.diff(a[1]) >= 0)
        assert np.all(np.diff(a[2]) >= 0)
        for tree in a[0]:
            validate_tree(tree)


def test_separate_budget_runs_agree():
    rng, p, y, c, tag, inp = fixture(66)
    u = rng.random((3, 16, 2))
    tree = SEED_TREE.copy(); tree[6:] = np.array([0, 9, 12, 21])
    both = tree_repair(tree, p, y, c, tag, inp, u, np.array([4, 16]))
    for i, budget in enumerate([4, 16]):
        one = tree_repair(tree, p, y, c, tag, inp, u, np.array([budget]))
        np.testing.assert_array_equal(both[:, i], one[:, 0])


def test_repair_monotone_against_initial_and_pure():
    rng, p, y, c, tag, inp = fixture(77)
    u = rng.random((3, 16, 2))
    old = p.copy()
    scores = tree_repair(SEED_TREE, p, y, c, tag, inp, u, np.array([16]))
    for i in range(3):
        initial = 64 - int(int(truth(p[i], inp)) ^ int(y[i])).bit_count()
        assert initial <= scores[i, 0] <= 64
    np.testing.assert_array_equal(p, old)


def test_noop_proposals_counted():
    rng, p, y, c, tag, inp = fixture(88)
    # Seed uniform author chooses first feature gene, resets it to its current 0.
    ut = rng.random((3, 8, 2))
    um = np.zeros((8, 2)); ur = rng.random((8, 10))
    for regime in [0, 1]:
        snapshots, _, accepted, diag = develop_tree(
            regime, p, y, c, tag, inp, ut, um, ur, np.array([8]), 8)
        np.testing.assert_array_equal(snapshots[0], SEED_TREE)
        assert accepted[0] == 0
        assert diag[0, 2] == 8
