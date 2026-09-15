import numpy as np
from src.programs import (popcount, truth, input_tables, circuit_schema, make_tasks,
                          repair_scores, develop, SEED_POLICY, POLICY_CARDS)


def test_packed_boolean_semantics():
    inputs = input_tables()
    for x in [0, 1, 2, 2**63, 2**64-1, 123456789]:
        assert popcount(np.uint64(x)) == x.bit_count()
    for op in range(4):
        circuit = np.array([op, 0, 1, 6], dtype=np.int64)
        word = int(truth(circuit, inputs))
        for x in range(64):
            a, b = x & 1, (x >> 1) & 1
            target = [a & b, a | b, a ^ b, 1-(a & b)][op]
            assert (word >> x) & 1 == target


def test_monotone_release_and_emulation():
    rng = np.random.default_rng(23)
    parents, targets = make_tasks(rng, 3, 8, 'mixed')
    cards, tags = circuit_schema(8); inputs = input_tables()
    u = rng.uniform(size=(3, 8, 3))
    budgets = np.array([1, 4, 8])
    scores = repair_scores(SEED_POLICY, parents, targets, cards, tags, inputs, u, budgets)
    assert np.all(np.diff(scores.astype(int), axis=1) >= 0)
    args = (parents, targets, cards, tags, inputs, u,
            rng.uniform(size=(8, 3)), rng.uniform(size=(8, 8)), np.array([2, 8]), 8)
    r = develop(0, *args); emulated = develop(3, *args)
    assert all(np.array_equal(a, b) for a, b in zip(r, emulated))
    for mode in [0, 1, 2]:
        policies, dev_scores, accepted = develop(mode, *args)
        assert np.all(policies >= 0) and np.all(policies < POLICY_CARDS)
        assert np.all(np.diff(dev_scores) >= 0)
        assert np.all(np.diff(accepted) >= 0)


def test_random_circuits_against_scalar_reference_and_input_immutability():
    """Independent scalar interpreter checks all 64 inputs of 100 circuits."""
    rng=np.random.default_rng(192)
    cards,tags=circuit_schema(16)
    for _ in range(100):
        c=np.array([rng.integers(int(k)) for k in cards],dtype=np.int64)
        word=int(truth(c,input_tables()))
        for x in range(64):
            values=[bool((x>>j)&1) for j in range(6)]
            for j in range(16):
                op,a,b=map(int,c[3*j:3*j+3]); left,right=values[a],values[b]
                value=(left and right) if op==0 else ((left or right) if op==1 else ((left!=right) if op==2 else not(left and right)))
                values.append(value)
            assert ((word>>x)&1)==int(values[int(c[-1])])
    parents,targets=make_tasks(rng,3,16,'xor')
    original=parents.copy(); targetcopy=targets.copy(); policy=SEED_POLICY.copy()
    repair_scores(policy,parents,targets,cards,tags,input_tables(),rng.uniform(size=(3,8,3)),np.array([8]))
    assert np.array_equal(parents,original) and np.array_equal(targets,targetcopy)
    assert np.array_equal(policy,SEED_POLICY)
