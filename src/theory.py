"""Finite-state audits. Rewards in [0,1]; TV(p,q)=sum(abs(p-q))/2.

Rectangular robustness is NOT sharp for a shared unknown parameter.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def probability_vector(p: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.min(p) < -1e-12:
        raise ValueError('Expected a finite nonnegative vector')
    if not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError('Probabilities must sum to one')
    return np.maximum(p, 0.0) / np.maximum(p, 0.0).sum()


def tv_extreme(p: ArrayLike, value: ArrayLike, radius: float,
               maximize: bool = False, allowed: ArrayLike | None = None
               ) -> tuple[float, NDArray[np.float64]]:
    """Exact linear optimization on a simplex intersected with a TV ball.

    Greedily transport from high-value to low-value coordinates. The optional
    mask specifies structural zeros, not merely zeros observed in samples.
    """
    p = probability_vector(p)
    v = np.asarray(value, dtype=float)
    if v.shape != p.shape or not np.all(np.isfinite(v)):
        raise ValueError('Finite values must have the same shape as p')
    if not 0 <= radius <= 1:
        raise ValueError('TV radius must lie in [0,1]')
    mask = np.ones(len(p), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if mask.shape != p.shape or not mask.any() or np.any(p[~mask] > 1e-12):
        raise ValueError('Invalid structural support')
    objective = -v if maximize else v
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(objective[idx], kind='stable')]
    q = p.copy()
    lo, hi, remaining = 0, len(order) - 1, float(radius)
    while lo < hi and remaining > 1e-14:
        a, b = order[lo], order[hi]
        if objective[a] >= objective[b] - 1e-14:
            break
        mass = min(remaining, 1.0 - q[a], q[b])
        q[a] += mass
        q[b] -= mass
        remaining -= mass
        if q[a] >= 1.0 - 1e-14:
            lo += 1
        if q[b] <= 1e-14:
            hi -= 1
    return float(q @ v), q


@dataclass
class RobustValue:
    lower: float
    upper: float
    lower_values: NDArray[np.float64]
    upper_values: NDArray[np.float64]
    lower_kernels: NDArray[np.float64]
    upper_kernels: NDArray[np.float64]


def robust_value(kernels: ArrayLike, reward: ArrayLike, initial: ArrayLike,
                 radii: ArrayLike, support: ArrayLike | None = None) -> RobustValue:
    """Sharp interval for row/time/arm-rectangular TV uncertainty.

    kernels: (h,s,s); radii: scalar or broadcastable to (h,s).
    Includes attained extremal kernels to verify the answer independently.
    """
    k = np.asarray(kernels, dtype=float)
    if k.ndim != 3 or k.shape[1] != k.shape[2]:
        raise ValueError('Expected kernels of shape (h,s,s)')
    h, s, _ = k.shape
    mu = probability_vector(initial)
    r = np.asarray(reward, dtype=float)
    if len(mu) != s or r.shape != (s,) or not np.all(np.isfinite(r)) or np.any((r < 0) | (r > 1)):
        raise ValueError('Invalid initial distribution or terminal reward')
    eps = np.broadcast_to(np.asarray(radii, dtype=float), (h, s))
    masks = np.ones_like(k, dtype=bool) if support is None else np.broadcast_to(np.asarray(support, dtype=bool), k.shape)
    lower = np.empty((h + 1, s)); upper = np.empty_like(lower)
    kl = np.empty_like(k); ku = np.empty_like(k)
    lower[h] = upper[h] = r
    for t in range(h - 1, -1, -1):
        for x in range(s):
            lower[t, x], kl[t, x] = tv_extreme(k[t, x], lower[t + 1], eps[t, x], False, masks[t, x])
            upper[t, x], ku[t, x] = tv_extreme(k[t, x], upper[t + 1], eps[t, x], True, masks[t, x])
    return RobustValue(float(mu @ lower[0]), float(mu @ upper[0]), lower, upper, kl, ku)


def evaluate(kernels: ArrayLike, reward: ArrayLike, initial: ArrayLike) -> float:
    distribution = probability_vector(initial).copy()
    for k in np.asarray(kernels, dtype=float):
        if k.shape != (len(distribution), len(distribution)):
            raise ValueError('Invalid transition shape')
        if np.min(k) < -1e-12 or not np.allclose(k.sum(axis=1), 1):
            raise ValueError('Invalid stochastic transition')
        distribution = distribution @ k
    return float(distribution @ np.asarray(reward, dtype=float))


def development_decomposition(recursive: ArrayLike, fixed: ArrayLike,
                              terminal_value: ArrayLike, initial: ArrayLike
                              ) -> tuple[float, NDArray[np.float64]]:
    """Established performance-difference identity, specialized to inheritance."""
    pr, pf = np.asarray(recursive, float), np.asarray(fixed, float)
    if pr.shape != pf.shape or pr.ndim != 3:
        raise ValueError('Matching transition sequences required')
    h, s, _ = pf.shape
    vf = np.empty((h + 1, s)); vf[h] = np.asarray(terminal_value, float)
    for t in range(h - 1, -1, -1):
        vf[t] = pf[t] @ vf[t + 1]
    dist = probability_vector(initial).copy()
    terms = np.empty(h)
    for t in range(h):
        terms[t] = dist @ ((pr[t] - pf[t]) @ vf[t + 1])
        dist = dist @ pr[t]
    return float(terms.sum()), terms


def hoeffding_interval(differences: ArrayLike, alpha: float = .05,
                       comparisons: int = 1, bias: float = 0.0
                       ) -> tuple[float, float, float]:
    """Input unit is an independent development-pair average, NOT a descendant.

    bias bounds absolute distortion of the CONTRAST, not one branch.
    """
    z = np.asarray(differences, dtype=float)
    if z.ndim != 1 or len(z) < 1 or np.any(np.abs(z) > 1 + 1e-12) or not np.all(np.isfinite(z)):
        raise ValueError('Independent bounded development-pair averages required')
    if not 0 < alpha < 1 or comparisons < 1 or bias < 0:
        raise ValueError('Invalid confidence parameters')
    mean = float(z.mean())
    radius = math.sqrt(2 * math.log(2 * comparisons / alpha) / len(z)) + bias
    return mean, max(-1., mean - radius), min(1., mean + radius)


def endpoint_interval(observed_r: float, observed_f: float,
                      beta_r: float, beta_f: float) -> tuple[float, float]:
    """Sharp Bernoulli-mean interval with independent arm distortion bounds."""
    if not all(0 <= x <= 1 for x in [observed_r, observed_f, beta_r, beta_f]):
        raise ValueError('Means and branch radii must lie in [0,1]')
    return (max(0., observed_r - beta_r) - min(1., observed_f + beta_f),
            min(1., observed_r + beta_r) - max(0., observed_f - beta_f))


def optimal_inner(development_cost: float, audit_cost: float,
                  between_variance: float, within_variance: float) -> float:
    """Continuous optimum for variance (tau2+sigma2/m)/n and cost n*(cd+m*ca)."""
    if development_cost <= 0 or audit_cost <= 0 or between_variance <= 0 or within_variance < 0:
        raise ValueError('Positive costs/between variance required')
    return max(1., math.sqrt(development_cost * within_variance / (audit_cost * between_variance)))


def empirical_bernstein_interval(differences: ArrayLike, alpha: float = .05,
                                 comparisons: int = 1, bias: float = 0.0
                                 ) -> tuple[float, float, float]:
    """Two-sided finite-sample Maurer--Pontil (2009, Thm 4) interval.

    Apply the one-sided bound to (Z+1)/2 and its complement; union over
    `comparisons` prespecified contrasts. Requires IID development-pair
    averages in [-1,1], n>=2. This is not a confidence sequence.
    """
    z = np.asarray(differences, dtype=float)
    if z.ndim != 1 or len(z) < 2 or not np.all(np.isfinite(z)) or np.any(np.abs(z) > 1+1e-12):
        raise ValueError('At least two IID bounded development-pair averages required')
    if not 0 < alpha < 1 or comparisons < 1 or bias < 0:
        raise ValueError('Invalid confidence parameters')
    logterm = math.log(4 * comparisons / alpha)
    radius = (math.sqrt(2 * float(z.var(ddof=1)) * logterm / len(z))
              + 14 * logterm / (3 * (len(z)-1)) + bias)
    mean = float(z.mean())
    return mean, max(-1., mean-radius), min(1., mean+radius)


def shared_history_interval(history: ArrayLike, values_r: ArrayLike,
                            values_f: ArrayLike, radius: float
                            ) -> tuple[float, float]:
    """Sharp contrast interval for ONE common uncertain history distribution.

    Unlike independent-arm rectangular uncertainty, a shared distribution
    acts on the conditional contrast. A constant contrast is invariant.
    """
    p = probability_vector(history)
    r, f = np.asarray(values_r, float), np.asarray(values_f, float)
    if r.shape != p.shape or f.shape != p.shape or np.any((r<0)|(r>1)) or np.any((f<0)|(f>1)):
        raise ValueError('Conditional branch values must lie in [0,1]')
    return tv_extreme(p, r-f, radius)[0], tv_extreme(p, r-f, radius, True)[0]
