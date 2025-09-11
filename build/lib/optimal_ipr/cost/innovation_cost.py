from __future__ import annotations
import numpy as np
from typing import Callable

def _expect_power_term(f: Callable[[np.ndarray], np.ndarray],
                       gamma_c: float,
                       n_grid_points: int = 20001) -> float:
    """Compute E[(1-theta)**gamma_c] under density f on [0,1]."""
    theta = np.linspace(0.0, 1.0, n_grid_points)
    pdf = np.clip(f(theta), 0.0, np.inf)
    # normalize defensively to 1 if f is slightly off due to numerics
    area = np.trapz(pdf, theta)
    if area <= 0.0:
        raise ValueError("Density integrates to zero or negative over [0,1].")
    pdf /= area
    term = np.power(1.0 - theta, gamma_c)
    return float(np.trapz(term * pdf, theta))

def build_cost_function(
    f: Callable[[np.ndarray], np.ndarray],
    TARGET_AVG_COST_SHARE: float,
    C_MIN_COST: float,
    GAMMA_C_COST: float,
) -> Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]:
    """
    Calibrate kappa_c from: TARGET_AVG_COST_SHARE = C_MIN_COST + kappa_c * E[(1-theta)**GAMMA_C_COST]
    Return c(theta, v) = v * (C_MIN_COST + kappa_c * (1-theta)**GAMMA_C_COST).
    """
    if TARGET_AVG_COST_SHARE < C_MIN_COST:
        raise ValueError("TARGET_AVG_COST_SHARE must be >= C_MIN_COST.")
    if GAMMA_C_COST < 0:
        raise ValueError("GAMMA_C_COST must be >= 0.")

    exp_term = _expect_power_term(f, GAMMA_C_COST)
    if exp_term <= 0.0:
        raise ValueError("Expectation E[(1-theta)**gamma] is non-positive.")

    kappa_c = (TARGET_AVG_COST_SHARE - C_MIN_COST) / exp_term

    def c(theta, v):
        theta_arr = np.asarray(theta, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        theta_arr = np.clip(theta_arr, 0.0, 1.0)
        cost_share = C_MIN_COST + kappa_c * np.power(1.0 - theta_arr, GAMMA_C_COST)
        return cost_share * v_arr

    return c
