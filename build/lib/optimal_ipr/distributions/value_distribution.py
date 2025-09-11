import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

def _max_multiple(sigma: float, N: int = 3_300_000, target_mean: float = 1.0) -> float:
    mu = np.log(target_mean) - 0.5 * sigma**2
    z = norm.ppf(1 - 1.0 / N)
    return np.exp(mu + sigma * z) / target_mean

def _sigma_for_target_max(M: float, n_v: int) -> float:
    x, _ = hermgauss(n_v)
    zmax = np.sqrt(2.0) * x.max()
    disc = zmax**2 - 2.0 * np.log(M)
    if disc < 0:
        raise ValueError("M too large for chosen n_v; increase n_v or lower sigma.")
    return zmax - np.sqrt(disc)

def _v_dist(n_v: int, sigma_log: float, target_mean: float = 1.0):
    nodes, w = hermgauss(n_v)
    z = np.sqrt(2.0) * nodes
    mu_log = np.log(target_mean) - 0.5 * sigma_log**2
    v_grid = np.exp(mu_log + sigma_log * z)
    v_weight = w / np.sqrt(np.pi)
    return v_grid, v_weight

def value_distribution(n_v: int = 51, sigma: float = 3.0, target_mean: float = 1.0):
    """Return (v_grid, v_weight) using Gauss–Hermite with literature sigma."""
    M = _max_multiple(sigma, target_mean=target_mean)
    sigma_log = _sigma_for_target_max(M, n_v)
    return _v_dist(n_v, sigma_log, target_mean)
