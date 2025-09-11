from __future__ import annotations
import numpy as np
from typing import Callable

# Match the .py: use total market size for the upper bound on perceived competitors
_N_TOTAL_FIRMS = 29_990  # max_competitors

class PerceivedProbability:
    def __init__(
        self,
        agent_types_grid: np.ndarray,
        F_func: Callable[[np.ndarray], np.ndarray],
        inverse_F_func: Callable[[np.ndarray], np.ndarray],
        base_k: float,
        max_competitors: int,
        min_competitors: int,
    ):
        self._agent_types = agent_types_grid
        self._F = F_func
        self._F_inv = inverse_F_func
        self._base_k = float(base_k)
        self._max_n = int(max_competitors)
        self._min_n = int(min_competitors)
        self._cache = {}  # cache full p(theta,·) curve per v

    def _get_perceived_n(self, theta: float) -> int:
        # Linear map: F(theta)=0 -> max_n; F(theta)=1 -> min_n
        n = self._max_n - (self._max_n - self._min_n) * float(self._F(np.array([theta]))[0])
        return int(max(n, self._min_n))

    def _generate_curve(self, v: float) -> np.ndarray:
        # One-time construction of p(theta,v) over theta grid, cached by v
        thetas = self._agent_types
        probs = np.zeros_like(thetas, dtype=float)
        k_eff = self._base_k * float(v)

        for i, theta in enumerate(thetas):
            n_perceived = self._get_perceived_n(theta)

            # perceived market from current type upward
            start_q = float(self._F(np.array([max(0.0, theta)]))[0])
            qs = np.linspace(start_q, 1.0, n_perceived)
            types_perceived = np.clip(self._F_inv(qs), 0.0, 1.0)

            # log-sum-exp for numerical stability
            log_my = k_eff * np.log1p(theta)
            log_comp = k_eff * np.log1p(types_perceived)
            all_logs = np.concatenate(([log_my], log_comp))
            m = np.max(all_logs)
            log_den = m + np.log(np.exp(all_logs - m).sum())
            probs[i] = np.exp(log_my - log_den)

        return probs

    def get_prob(self, theta: np.ndarray | float, v: float) -> np.ndarray | float:
        # Cache per v; interpolate over theta grid
        v_key = float(v)
        if v_key not in self._cache:
            self._cache[v_key] = self._generate_curve(v_key)
        return np.interp(np.asarray(theta), self._agent_types, self._cache[v_key])

def build_subjective_probability(
    base_k: float,
    m_comp: int,
    F: Callable[[np.ndarray], np.ndarray],
    F_inv: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray | float, float], np.ndarray | float]:
    """
    Inputs: base_k, m_comp (minimum competitors), F, F_inv.
    Output: p(theta, v) as a callable.
    """
    theta_grid = np.linspace(0.0, 1.0, 500)
    model = PerceivedProbability(
        agent_types_grid=theta_grid,
        F_func=F,
        inverse_F_func=F_inv,
        base_k=base_k,
        max_competitors=_N_TOTAL_FIRMS,
        min_competitors=m_comp,
    )
    return model.get_prob