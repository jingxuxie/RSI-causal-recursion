import itertools
import numpy as np
import pytest
from src.shared_adapter import (persistent_values, persistent_interval,
                                rectangular_interval, consistency_example)


@pytest.mark.parametrize('gap', [.01, .1, .2, .7, 1.])
def test_persistence_counterexample(gap):
    model = consistency_example(gap)
    np.testing.assert_allclose(persistent_values(*model), [gap, gap])
    np.testing.assert_allclose(persistent_interval(*model), [gap, gap])
    np.testing.assert_allclose(rectangular_interval(*model), [-gap, gap])


def test_rectangular_against_bruteforce():
    rng = np.random.default_rng(719)
    for _ in range(10):
        k = rng.dirichlet(np.ones(2), size=(2, 2, 2))
        initial = rng.dirichlet(np.ones(2))
        reward = rng.uniform(-1, 1, 2)
        values = []
        for choices in itertools.product(range(2), repeat=4):
            d = initial.copy()
            for t in range(2):
                transition = np.array([k[choices[t*2+s], t, s] for s in range(2)])
                d = d @ transition
            values.append(d @ reward)
        np.testing.assert_allclose(rectangular_interval(k, initial, reward), [min(values), max(values)])


def test_persistent_simulation_matches_exact():
    rng = np.random.default_rng(371)
    k = rng.dirichlet(np.ones(3), size=(3, 4, 3))
    initial = rng.dirichlet(np.ones(3)); reward = np.array([-.5, .1, .9])
    expected = persistent_values(k, initial, reward)
    for a in range(3):
        d = initial.copy()
        for t in range(4):
            d = d @ k[a, t]
        np.testing.assert_allclose(d @ reward, expected[a])
    lo, hi = rectangular_interval(k, initial, reward)
    assert lo <= expected.min() <= expected.max() <= hi


def test_sharp_mixture_endpoints_and_interior():
    rng = np.random.default_rng(551)
    k = rng.dirichlet(np.ones(3), size=(3, 4, 3))
    nu = np.array([1., 0, 0]); r = np.array([0., .5, 1.])
    v = persistent_values(k, nu, r)
    lo, hi = persistent_interval(k, nu, r)
    for mix in np.linspace(0, 1, 11):
        law = np.zeros(3); law[v.argmin()] += 1-mix; law[v.argmax()] += mix
        np.testing.assert_allclose(law @ v, (1-mix)*lo+mix*hi)


def test_invalid_inputs():
    k, nu, r = consistency_example()
    k[0, 0, 0, 0] = -.1
    with pytest.raises(ValueError):
        persistent_values(k, nu, r)
