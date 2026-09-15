import numpy as np
import pytest
from scipy.optimize import linprog
from src.theory import (tv_extreme, robust_value, evaluate, development_decomposition,
                        hoeffding_interval, endpoint_interval, optimal_inner)


def linear_program_extreme(p, v, eps, maximize=False):
    n = len(p)
    # Variables q and absolute-deviation auxiliaries d.
    c = np.r_[(-v if maximize else v), np.zeros(n)]
    aub = np.vstack([np.c_[np.eye(n), -np.eye(n)],
                     np.c_[-np.eye(n), -np.eye(n)],
                     np.r_[np.zeros(n), np.ones(n)][None, :]])
    bub = np.r_[p, -p, 2*eps]
    ae = np.r_[np.ones(n), np.zeros(n)][None, :]
    ans = linprog(c, A_ub=aub, b_ub=bub, A_eq=ae, b_eq=[1], bounds=(0, None), method='highs')
    assert ans.success
    return float(ans.x[:n] @ v)


@pytest.mark.parametrize('seed', range(20))
def test_tv_against_independent_lp(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 12)); p = rng.dirichlet(np.ones(n))
    v = rng.normal(size=n); eps = float(rng.uniform())
    for maximize in [False, True]:
        value, q = tv_extreme(p, v, eps, maximize)
        assert np.all(q >= -1e-12) and abs(q.sum()-1) < 1e-10
        assert np.abs(q-p).sum()/2 <= eps+1e-10
        assert value == pytest.approx(linear_program_extreme(p, v, eps, maximize), abs=1e-8)


@pytest.mark.parametrize('seed', range(8))
def test_robust_attained_and_decomposition(seed):
    rng = np.random.default_rng(seed)
    h, s = 5, 4
    kr = rng.dirichlet(np.ones(s), size=(h, s))
    kf = rng.dirichlet(np.ones(s), size=(h, s))
    r = rng.uniform(size=s); mu = rng.dirichlet(np.ones(s))
    ans = robust_value(kr, r, mu, .08)
    nominal = evaluate(kr, r, mu)
    assert ans.lower <= nominal <= ans.upper
    assert evaluate(ans.lower_kernels, r, mu) == pytest.approx(ans.lower)
    assert evaluate(ans.upper_kernels, r, mu) == pytest.approx(ans.upper)
    zero = robust_value(kr, r, mu, 0)
    assert zero.lower == pytest.approx(nominal)
    assert zero.upper == pytest.approx(nominal)
    total, terms = development_decomposition(kr, kf, r, mu)
    assert total == pytest.approx(nominal - evaluate(kf, r, mu), abs=1e-12)


def test_structural_support_and_constants():
    v, q = tv_extreme([.2, .8, 0], [1, 1, -100], 1, allowed=[1, 1, 0])
    assert v == pytest.approx(1)
    assert q[2] == 0
    with pytest.raises(ValueError):
        tv_extreme([.2, .8], [0, 1], -1)
    with pytest.raises(ValueError):
        tv_extreme([.2, .8], [0, 1], .1, allowed=[1, 0])


def test_intervals_and_nesting():
    assert endpoint_interval(.6, .4, .1, .1) == pytest.approx((0, .4))
    mean, lo, hi = hoeffding_interval(np.ones(10000)*.2)
    assert 0 < lo < mean < hi
    assert optimal_inner(100, 1, .01, .25) == 50


def test_empirical_bernstein_and_shared_history():
    from src.theory import empirical_bernstein_interval, shared_history_interval
    z = np.linspace(-.2,.2,100)
    mean, lo, hi = empirical_bernstein_interval(z, comparisons=3)
    logterm=np.log(4*3/.05)
    expected=np.sqrt(2*z.var(ddof=1)*logterm/100)+14*logterm/(3*99)
    assert np.isclose(mean, 0) and np.isclose(hi,expected) and np.isclose(lo,-expected)
    assert shared_history_interval([.5,.5],[.3,.9],[.2,.8],1)==pytest.approx((.1,.1))
    with pytest.raises(ValueError):
        empirical_bernstein_interval([0])
    with pytest.raises(ValueError):
        empirical_bernstein_interval([0,2])


def test_endpoint_overlap_and_kl_separation():
    from scipy.special import rel_entr
    for b in [.01,.03,.10]:
        gamma=1.5*b
        obs=np.array([.5+gamma/4,.5-gamma/4])
        for ideal in [np.array([.5,.5]),np.array([.5+gamma/2,.5-gamma/2])]:
            assert np.max(np.abs(obs-ideal))<=b/2+1e-12
        for gamma in np.linspace(2*b+.001,.25,7):
            gap=gamma-2*b
            p0=np.array([.5+b/2,.5-b/2]);p1=np.array([.5+(gamma-b)/2,.5-(gamma-b)/2])
            divergence=np.sum(rel_entr(p1,p0)+rel_entr(1-p1,1-p0))
            assert divergence<=4*gap**2+1e-12
