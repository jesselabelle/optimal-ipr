from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import powerlaw

# Constants match the notebook/.py
_DATA_FILENAME = "data/top2000_SB2024.xlsx"
_N_TOTAL_FIRMS = 29_990                 # firms
_TOTAL_RD_SPENDING_MUSD = 841_141       # millions USD
_EUR_USD = 1.09
_SEED = 42

def _resolve_data_path() -> Path:
    # try cwd/data/..., then package-root/data/...
    cwd = Path.cwd()
    p1 = cwd / _DATA_FILENAME
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

def _type_distribution(data_path: Path,
                       total_firms: int,
                       total_rd_spending_musd: float,
                       eur_usd_rate: float = _EUR_USD):
    # Tail from scorecard (US firms), column index matches the notebook
    df = pd.read_excel(data_path)
    df = df[df['Country'] == 'US'].copy()
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

    # Simple lognormal for body with sigma chosen as in notebook spirit.
    # If body_mean<=0, make an empty body.
    rng = np.random.default_rng(_SEED)
    if N_body > 0 and body_mean > 0:
        # conservative body sigma; the notebook estimates it, but we keep mass and mean intact
        body_sigma = 0.8
        body_mu = np.log(body_mean) - 0.5 * body_sigma**2
        S_body = rng.lognormal(mean=body_mu, sigma=body_sigma, size=N_body)
    else:
        S_body = np.array([], dtype=float)

    # Combine body + tail spending
    S = np.sort(np.concatenate([S_body, S_tail]))
    N = len(S)

    # Fit Pareto tail from xmin_tail, like the notebook
    fit = powerlaw.Fit(S, discrete=False, xmin=xmin_tail)
    alpha = float(fit.alpha)
    xmin = float(fit.xmin)
    p_tail = float((S >= xmin).sum() / N)

    # Map spending to theta with s0 = xmin (then theta_u = 0.5)
    s0 = xmin
    theta = S / (S + s0)
    theta = np.sort(theta)
    theta_u = 0.5
    mask_tail = (S >= xmin)
    theta_body = np.sort(theta[~mask_tail])
    q_body = 1.0 - p_tail

    # Empirical CDF on theta (deterministic mapping)
    def F_empirical(t: float) -> float:
        # right-continuous ECDF
        idx = np.searchsorted(theta, t, side="right")
        return idx / N

    details = dict(alpha=alpha, xmin=xmin, p_tail=p_tail, q_body=q_body,
                   theta_u=theta_u, s0=s0, N=N)
    return F_empirical, theta, theta_body, fit, details

def _hybrid_continuous_from_noise(theta_body: np.ndarray,
                                  fit: powerlaw.Fit,
                                  details: dict,
                                  noise_level: float,
                                  n_grid_points: int = 20001
                                  ) -> Tuple[Callable[[np.ndarray], np.ndarray],
                                             Callable[[np.ndarray], np.ndarray]]:
    """Construct f and F on [0,1] using body KDE + Pareto tail in theta-space."""
    alpha = details["alpha"]; theta_u = details["theta_u"]
    p_tail = details["p_tail"]; q_body = details["q_body"]

    # Add lognormal noise to body thetas by multiplicative shock to spending mapping proxy.
    # Operate in logit(theta): log(theta/(1-theta)), add Normal(0, sigma), map back.
    # This keeps support in (0,1) and mirrors “lognormal-like” noise.
    def _logit(x): return np.log(x) - np.log1p(-x)
    def _inv_logit(z): return 1.0 / (1.0 + np.exp(-z))

    if theta_body.size > 0:
        z = _logit(np.clip(theta_body, 1e-9, 1 - 1e-9))
        z_noisy = z + np.random.default_rng(_SEED).normal(0.0, noise_level, size=z.shape)
        theta_body_noisy = _inv_logit(z_noisy)
    else:
        theta_body_noisy = theta_body

    # Grid
    theta_grid = np.linspace(0.0, 1.0, n_grid_points)
    body_mask = theta_grid < theta_u
    tail_mask = ~body_mask

    # BODY: KDE on [0, theta_u), then scale to mass q_body
    if theta_body_noisy.size >= 5:
        kde = gaussian_kde(theta_body_noisy, bw_method="scott")
        kde_pdf_raw = np.where(body_mask, kde(theta_grid), 0.0)
        mass_raw_body = np.trapz(kde_pdf_raw[body_mask], theta_grid[body_mask])
        kde_pdf = (kde_pdf_raw / mass_raw_body) * q_body if mass_raw_body > 0 else 0.0 * kde_pdf_raw
    else:
        kde_pdf = np.zeros_like(theta_grid)

    # TAIL: exact Pareto transform in theta-space, scaled to p_tail
    with np.errstate(divide="ignore", invalid="ignore"):
        tail_pdf_shape = (alpha - 1.0) * (1.0 - theta_grid) ** (alpha - 2.0) * (theta_grid ** (-alpha))
        tail_pdf_shape = np.nan_to_num(tail_pdf_shape, nan=0.0, posinf=0.0, neginf=0.0)
    # zero out body part
    tail_pdf = np.where(tail_mask, tail_pdf_shape, 0.0)
    # scale tail to integrate to p_tail
    mass_raw_tail = np.trapz(tail_pdf[tail_mask], theta_grid[tail_mask])
    tail_pdf = (tail_pdf / mass_raw_tail) * p_tail if mass_raw_tail > 0 else 0.0 * tail_pdf

    f_vals = kde_pdf + tail_pdf
    # numerical CDF with clipping
    F_vals = np.cumsum(f_vals) * (theta_grid[1] - theta_grid[0])
    F_vals = np.clip(F_vals, 0.0, 1.0)

    # f(theta) and F(theta) as callables
    def f(theta_in: np.ndarray) -> np.ndarray:
        return np.interp(theta_in, theta_grid, f_vals, left=0.0, right=0.0)

    def F(theta_in: np.ndarray) -> np.ndarray:
        return np.interp(theta_in, theta_grid, F_vals, left=0.0, right=1.0)

    return f, F

def build_theta_distribution(noise_level: float,
                             n_grid_points: int = 20001
                             ) -> Tuple[Callable[[np.ndarray], np.ndarray],
                                        Callable[[np.ndarray], np.ndarray],
                                        Callable[[np.ndarray], np.ndarray]]:
    """
    Input: noise_level.
    Output: (f, F, F_inv) where each is vectorized on [0,1].
    All other assumptions mirror the notebook/.py.
    """
    data_path = _resolve_data_path()
    F_emp, theta_all, theta_body, fit, details = _type_distribution(
        data_path=data_path,
        total_firms=_N_TOTAL_FIRMS,
        total_rd_spending_musd=_TOTAL_RD_SPENDING_MUSD,
        eur_usd_rate=_EUR_USD,
    )
    f, F = _hybrid_continuous_from_noise(theta_body, fit, details, noise_level, n_grid_points)

    # Inverse CDF via interpolation on a dense grid
    interp_grid = np.linspace(0.0, 1.0, 10000)
    cdf_vals = F(interp_grid)
    F_inv = interp1d(cdf_vals, interp_grid, bounds_error=False, fill_value=(0.0, 1.0))
    return f, F, F_inv
