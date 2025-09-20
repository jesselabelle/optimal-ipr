from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple

import powerlaw
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, norm

# Constants match the notebook/.py
_DATA_FILENAME = "data/top2000_SB2024.xlsx"
_N_TOTAL_FIRMS = 29_990  # firms
_TOTAL_RD_SPENDING_MUSD = 841_141  # millions USD
_EUR_USD = 1.09
_SEED = 42


def _max_multiple(sigma: float, N: int, target_mean: float) -> float:
    mu = np.log(target_mean) - 0.5 * sigma**2
    z = norm.ppf(1 - 1.0 / N)
    return np.exp(mu + sigma * z) / target_mean


def _sigma_for_target_max(M: float, n_v: int) -> float:
    x, _ = hermgauss(n_v)
    zmax = np.sqrt(2.0) * x.max()
    disc = zmax**2 - 2.0 * np.log(M)
    if disc < 0:
        raise ValueError("Need different ratio, M too large")
    return zmax - np.sqrt(disc)


def _resolve_data_path() -> Path:
    # try cwd/data/..., then package-root/data/...
    cwd = Path.cwd()
    p1 = cwd.parent / "data" / _DATA_FILENAME
    if p1.exists():
        return p1
    # package root guess: src/optimal_ipr/distributions/ -> go up three, then data/...
    here = Path(__file__).resolve()
    root_guess = here.parents[3]
    p2 = root_guess / _DATA_FILENAME
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"R&D data file not found. Looked for '{p1}' and '{p2}'. "
        f"Place the Excel at project_root/{_DATA_FILENAME}."
    )


def _type_distribution(
    data_path: Path,
    total_firms: int,
    total_rd_spending_musd: float,
    eur_usd_rate: float = _EUR_USD,
):
    # Tail from scorecard (US firms), column index matches the notebook
    df = pd.read_excel(data_path)
    df = df[df["Country"] == "US"].copy()
    rd_col = df.columns[5]
    S_tail = pd.to_numeric(df[rd_col], errors="coerce").dropna()
    S_tail = S_tail[S_tail > 0] * eur_usd_rate  # EUR->USD
    S_tail = np.sort(S_tail.values.astype(float))

    N_tail = len(S_tail)
    xmin_tail = float(S_tail.min())

    # Body mass from totals (all in USD millions)
    N_body = int(total_firms - N_tail)
    T_tail = float(S_tail.sum())
    T_body = float(total_rd_spending_musd) - T_tail
    body_mean = T_body / N_body if N_body > 0 else 0.0

    rng = np.random.default_rng(_SEED)
    if N_body > 0 and body_mean > 0:
        mm = _max_multiple(0.57, N_body, body_mean)
        body_sigma = _sigma_for_target_max(mm, 8)
        body_mu = np.log(body_mean) - 0.5 * body_sigma**2
        S_body = rng.lognormal(body_mu, body_sigma, N_body)
    else:
        S_body = np.array([], dtype=float)

    # Combine body + tail spending
    S = np.sort(np.concatenate([S_body, S_tail]))
    N = len(S)
    total_spending = float(S.sum())
    # Fit Pareto tail from xmin_tail
    fit = powerlaw.Fit(S, discrete=False, xmin=xmin_tail)
    alpha = float(fit.alpha)
    xmin = float(fit.xmin)
    p_tail = float((S >= xmin).sum() / N)

    # Map spending to theta with s0 = xmin (theta_u = 0.5)
    s0 = xmin
    theta = S / (S + s0)
    theta_u = 0.5
    mask_tail = S >= xmin
    theta_body = np.sort(theta[~mask_tail])
    n_body = len(theta_body)
    q_body = 1.0 - p_tail

    def F_empirical(t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t < theta_u and n_body > 0:
            idx = np.searchsorted(theta_body, t, side="right")
            return q_body * (idx / n_body)
        if t < 1.0:
            survivor_share = p_tail * ((1.0 - t) / t) ** (alpha - 1.0)
            return 1.0 - survivor_share
        return 1.0

    details = {
        "p_tail": p_tail,
        "q_body": q_body,
        "theta_u": theta_u,
        "n_body": n_body,
        "theta_body": theta_body,
    }
    return F_empirical, theta, fit, details


def _stochastic_continuous_from_noise(
    theta_det: np.ndarray,
    fit: powerlaw.Fit,
    noise_level: float,
    n_grid_points: int = 20001,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Construct f and F on [0,1] using noisy theta data and Pareto tail."""

    rng = np.random.default_rng(_SEED)
    epsilon = rng.lognormal(mean=-(noise_level**2) / 2, sigma=noise_level, size=theta_det.shape)
    theta_stochastic = theta_det * epsilon
    theta_stochastic = np.clip(theta_stochastic, 0.0, 0.9999)
    theta_stochastic.sort()

    theta_u = 0.5
    q_body = float((theta_stochastic < theta_u).sum()) / theta_stochastic.size
    p_tail = 1.0 - q_body
    theta_body = theta_stochastic[theta_stochastic < theta_u]
    alpha = fit.alpha

    theta_grid = np.linspace(0.0, 1.0, n_grid_points)
    body_mask = theta_grid < theta_u

    if theta_body.size > 1:
        kde = gaussian_kde(theta_body, bw_method=0.1)
        kde_pdf_raw = kde(theta_grid)
        kde_pdf_raw[~body_mask] = 0.0
        mass_raw_body = np.trapz(kde_pdf_raw[body_mask], theta_grid[body_mask])
        kde_pdf = kde_pdf_raw * (q_body / mass_raw_body) if mass_raw_body > 0 else 0.0 * kde_pdf_raw
    else:
        kde_pdf = np.zeros_like(theta_grid)

    with np.errstate(divide="ignore", invalid="ignore"):
        tail_pdf = (
            p_tail * (alpha - 1.0) * (1.0 - theta_grid) ** (alpha - 2.0) * (theta_grid ** (-alpha))
        )
        tail_pdf = np.nan_to_num(tail_pdf, nan=0.0, posinf=0.0, neginf=0.0)

    f_vals = np.where(body_mask, kde_pdf, tail_pdf)
    f_vals[f_vals < 0] = 0.0
    f_vals /= np.trapz(f_vals, theta_grid)

    F_vals = np.cumsum(f_vals) * (theta_grid[1] - theta_grid[0])

    def f(theta_in: np.ndarray) -> np.ndarray:
        return np.interp(theta_in, theta_grid, f_vals, left=0.0, right=0.0)

    def F(theta_in: np.ndarray) -> np.ndarray:
        return np.interp(theta_in, theta_grid, F_vals, left=0.0, right=1.0)

    return f, F


def build_theta_distribution(noise_level: float, n_grid_points: int = 20001) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Input: noise_level.
    Output: (f, F, F_inv) where each function is vectorized on [0,1].
    All other assumptions mirror the notebook/.py.
    """
    data_path = _resolve_data_path()
    _, theta_det, fit, details = _type_distribution(
        data_path=data_path,
        total_firms=_N_TOTAL_FIRMS,
        total_rd_spending_musd=_TOTAL_RD_SPENDING_MUSD,
        eur_usd_rate=_EUR_USD,
    )
    f, F = _stochastic_continuous_from_noise(theta_det, fit, noise_level, n_grid_points)

    # Inverse CDF via interpolation on a dense grid
    interp_grid = np.linspace(0.0, 1.0, 10000)
    cdf_vals = F(interp_grid)
    F_inv = interp1d(cdf_vals, interp_grid, bounds_error=False, fill_value=(0.0, 1.0))

    return f, F, F_inv