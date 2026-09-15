"""Exact persistent-adapter audits and deliberately looser rectangular relaxations.

The adapter is shared by the two branches and is drawn ONCE per complete audit.
Kernels act on sufficient paired states. A rowwise relaxation may resample the
adapter at each time/state and therefore answers a different robustness question.
"""
from __future__ import annotations
import numpy as np


def validate(kernels: np.ndarray, initial: np.ndarray, reward: np.ndarray) -> None:
    if kernels.ndim != 4 or kernels.shape[2] != kernels.shape[3]:
        raise ValueError('Expected adapters x horizon x states x states')
    n = kernels.shape[2]
    if min(kernels.shape[:2]) < 1 or initial.shape != (n,) or reward.shape != (n,):
        raise ValueError('Empty model or inconsistent dimensions')
    if not all(np.all(np.isfinite(x)) for x in [kernels, initial, reward]):
        raise ValueError('All inputs must be finite')
    if np.any(kernels < 0) or not np.allclose(kernels.sum(axis=-1), 1, atol=1e-12):
        raise ValueError('Kernels must be stochastic')
    if np.any(initial < 0) or not np.isclose(initial.sum(), 1):
        raise ValueError('Invalid initial distribution')
    if np.any(np.abs(reward) > 1):
        raise ValueError('Paired terminal contrast must lie in [-1,1]')


def persistent_values(kernels: np.ndarray, initial: np.ndarray,
                      reward: np.ndarray) -> np.ndarray:
    validate(kernels, initial, reward)
    values = np.empty(kernels.shape[0])
    for adapter in range(len(kernels)):
        v = reward.copy()
        for t in range(kernels.shape[1] - 1, -1, -1):
            v = kernels[adapter, t] @ v
        values[adapter] = initial @ v
    return values


def persistent_interval(kernels: np.ndarray, initial: np.ndarray,
                        reward: np.ndarray) -> tuple[float, float]:
    """Sharp interval when arbitrary mixtures of persistent adapters are allowed.

    If the adapter is fixed but not mixed, the exact identified SET is the finite
    collection persistent_values; this interval is its sharp interval hull.
    """
    v = persistent_values(kernels, initial, reward)
    return float(v.min()), float(v.max())


def rectangular_interval(kernels: np.ndarray, initial: np.ndarray,
                         reward: np.ndarray) -> tuple[float, float]:
    """Relax the common persistent adapter to independent row choices."""
    validate(kernels, initial, reward)
    lo, hi = reward.copy(), reward.copy()
    for t in range(kernels.shape[1] - 1, -1, -1):
        lo = np.min(kernels[:, t] @ lo, axis=0)
        hi = np.max(kernels[:, t] @ hi, axis=0)
    return float(initial @ lo), float(initial @ hi)


def consistency_example(gap: float = .2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every persistent adapter gives +gap; illegal switching also permits -gap."""
    if not 0 < gap <= 1:
        raise ValueError('gap must lie in (0,1]')
    kernels = np.tile(np.eye(5), (2, 2, 1, 1))
    for adapter in range(2):
        kernels[adapter, 0, 0] = 0
        kernels[adapter, 0, 0, 1 + adapter] = 1
        for memory in range(2):
            kernels[adapter, 1, 1 + memory] = 0
            kernels[adapter, 1, 1 + memory, 3 if memory == adapter else 4] = 1
    initial = np.array([1., 0, 0, 0, 0])
    reward = np.array([0., 0, 0, gap, -gap])
    return kernels, initial, reward
