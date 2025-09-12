from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from tqdm.auto import tqdm

# Match the file constants
_N_TOTAL_FIRMS = 29_990  # N_total_firms


@njit
def _interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    n = xp.size
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    idx = np.searchsorted(xp, x) - 1
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@njit(parallel=True)
def _total_revenue_at_array(
    theta_tilde: float,
    theta_points: np.ndarray,
    theta_weights: np.ndarray,
    p_theta: np.ndarray,
    c_theta: np.ndarray,
    v: float,
    bar_beta: float,
    tau_d: float,
    n_total_firms: float,
    mass_of_investors: int,
    profit_shifted: bool,
) -> float:
    if mass_of_investors < 1:
        return 0.0
    imitator_profit = (1.0 - bar_beta) * v / mass_of_investors
    start_idx = np.searchsorted(theta_points, theta_tilde)
    total = 0.0
    for i in prange(start_idx, theta_points.size):
        if profit_shifted:
            innov_taxable = c_theta[i] * bar_beta
        else:
            innov_taxable = v * bar_beta
        win_prob = p_theta[i]
        expected_tax_base = win_prob * innov_taxable + (1.0 - win_prob) * imitator_profit
        total += tau_d * n_total_firms * expected_tax_base * theta_weights[i]
    return total


@njit
def _indifference_gap_array(
    theta_tilde: float,
    theta_points: np.ndarray,
    p_theta: np.ndarray,
    c_theta: np.ndarray,
    v: float,
    bar_beta: float,
    tau_d: float,
    tau_f: float,
    Z_val: float,
    profit_shifted: bool,
    mass_of_investors: float,
    transfer: float,
) -> float:
    c_th = _interp(theta_tilde, theta_points, c_theta)
    p_th = _interp(theta_tilde, theta_points, p_theta)
    if profit_shifted:
        innovator_payoff = bar_beta * ((1.0 - tau_f) * (v - c_th) + (1.0 - tau_d) * c_th)
    else:
        innovator_payoff = bar_beta * v * (1.0 - tau_d)
    if mass_of_investors >= 1.0:
        imitator_payoff = (1.0 - bar_beta) * v / mass_of_investors
    else:
        imitator_payoff = 0.0
    investing_utility = (
        p_th * (innovator_payoff - Z_val) + (1.0 - p_th) * imitator_payoff * (1.0 - tau_d) - c_th
    )
    return investing_utility - transfer


class InvestmentCutoffSolver:
    """
    Discrete-sum version as in In[14].
    """

    def __init__(
        self,
        p_func: Callable[[ArrayLike, float], np.ndarray],
        c_func: Callable[[ArrayLike, float], np.ndarray],
        Z_func: Callable[[ArrayLike, ArrayLike], np.ndarray],
        f_dist: Callable[[ArrayLike], np.ndarray],
        F_dist: Callable[[float], float],
        n_total_firms: float,
        tau_d: float,
        tau_f: float,
        theta_points: np.ndarray,
        theta_weights: np.ndarray,
    ):
        self.p, self.c, self.Z, self.f, self.F = p_func, c_func, Z_func, f_dist, F_dist
        self.n_total_firms = float(n_total_firms)
        self.tau_d, self.tau_f = float(tau_d), float(tau_f)
        self._profit_shifted = self.tau_d > self.tau_f

        self.theta_points = np.asarray(theta_points, dtype=float)
        self.theta_weights = np.asarray(theta_weights, dtype=float)

        self._v = None
        self._bar_beta = None
        self._rev_cache = {}
        self._p_theta = None
        self._c_theta = None
        self._Z_val = None

    def _total_revenue_at(
        self, theta_tilde: float, mass_of_investors: float | None = None
    ) -> float:
        key = float(theta_tilde)
        if key in self._rev_cache:
            return self._rev_cache[key]
        if mass_of_investors is None:
            mass_of_investors = int((1.0 - self.F(theta_tilde)) * self.n_total_firms)
        else:
            mass_of_investors = int(mass_of_investors)

        total_revenue = _total_revenue_at_array(
            theta_tilde,
            self.theta_points,
            self.theta_weights,
            self._p_theta,
            self._c_theta,
            self._v,
            self._bar_beta,
            self.tau_d,
            self.n_total_firms,
            mass_of_investors,
            self._profit_shifted,
        )
        self._rev_cache[key] = total_revenue
        return total_revenue

    def _get_transfer(
        self,
        theta_tilde: float,
        mass_of_investors: float,
        num_non_investors: int,
    ) -> float:
        if num_non_investors <= 1:
            return 0.0
        return self._total_revenue_at(theta_tilde, mass_of_investors) / num_non_investors

    def _indifference_gap(self, theta_tilde: float) -> float:
        th = float(theta_tilde)
        F_val = self.F(th)
        mass_of_investors = (1.0 - F_val) * self.n_total_firms
        num_non_investors = int(F_val * self.n_total_firms)
        transfer = self._get_transfer(th, mass_of_investors, num_non_investors)
        return _indifference_gap_array(
            th,
            self.theta_points,
            self._p_theta,
            self._c_theta,
            self._v,
            self._bar_beta,
            self.tau_d,
            self.tau_f,
            self._Z_val,
            self._profit_shifted,
            mass_of_investors,
            transfer,
        )

    def solve(self, v: float, bar_beta: float):
        self._v = float(v)
        self._bar_beta = float(bar_beta)
        self._rev_cache.clear()
        try:
            self._p_theta = np.asarray(self.p(self.theta_points, self._v), dtype=float)
        except Exception:
            self._p_theta = np.array([self.p(ti, self._v) for ti in self.theta_points])
        try:
            self._c_theta = np.asarray(self.c(self.theta_points, self._v), dtype=float)
        except Exception:
            self._c_theta = np.array([self.c(ti, self._v) for ti in self.theta_points])
        self._Z_val = float(self.Z(self._bar_beta, self._v))

        top = self._indifference_gap(0.999)
        if top < 0:
            return 1.0
        bot = self._indifference_gap(0.001)
        if bot > 0:
            return 0.0
        try:
            return brentq(self._indifference_gap, 0.001, 0.999, maxiter=200)
        except (ValueError, RuntimeError):
            return None


def choose_stochastic_winner(theta_tilde: float, v: float, p, theta_points: np.ndarray):
    """
    In[16]: randomly select one winner using conditional probabilities over θ ≥ θ̃.
    """
    if theta_tilde >= 1.0:
        return np.nan

    investor_mask = theta_points >= theta_tilde
    investor_thetas = theta_points[investor_mask]
    if len(investor_thetas) == 0:
        return np.nan

    investor_probs = p(investor_thetas, v)
    total_prob_mass = np.sum(investor_probs)
    if total_prob_mass <= 0:
        return np.nan

    conditional_win_probs = investor_probs / total_prob_mass
    conditional_win_probs = np.nan_to_num(conditional_win_probs)
    conditional_win_probs /= np.sum(conditional_win_probs)
    return np.random.choice(investor_thetas, p=conditional_win_probs)


def build_lookup_tables(
    p: Callable[[ArrayLike, float], np.ndarray],
    c: Callable[[ArrayLike, float], np.ndarray],
    Z: Callable[[ArrayLike, ArrayLike], np.ndarray],
    f: Callable[[ArrayLike], np.ndarray],
    F: Callable[[float], float],
    tau_d: ArrayLike,
    tau_f: ArrayLike,
    bar_beta: ArrayLike,
    v: ArrayLike,
    n_jobs: int | None = None,
):
    """
    Inputs: p, c, Z, f, F, tau_d (scalar or 1D grid), tau_f (scalar or 1D grid), bar_beta (1D grid), v (1D grid).
    Outputs: theta_tilde_table, theta_winner_table, and get_index functions.
    """
    # Build discrete theta quadrature to match In[9]
    theta_points = np.linspace(0.0, 1.0, 10000)
    theta_weights = f(theta_points)
    theta_weights = theta_weights / np.sum(theta_weights)

    tau_d_grid = np.atleast_1d(np.asarray(tau_d, dtype=float))
    tau_f_grid = np.atleast_1d(np.asarray(tau_f, dtype=float))
    bar_grid = np.atleast_1d(np.asarray(bar_beta, dtype=float))
    v_grid = np.atleast_1d(np.asarray(v, dtype=float))

    def compute_slice_for_taus(td, tf) -> Tuple[np.ndarray, np.ndarray]:
        """
        In[17]: produce both θ̃ and winner slices for one (tau_d, tau_f).
        """
        solver = InvestmentCutoffSolver(
            p, c, Z, f, F, _N_TOTAL_FIRMS, td, tf, theta_points, theta_weights
        )
        theta_tilde_slice = np.full((len(bar_grid), len(v_grid)), np.nan)
        theta_winner_slice = np.full((len(bar_grid), len(v_grid)), np.nan)

        for k, bar_beta_val in enumerate(bar_grid):
            for l, v_val in enumerate(v_grid):
                theta_tilde_val = solver.solve(v=v_val, bar_beta=bar_beta_val)
                if theta_tilde_val is not None and 0 < theta_tilde_val < 1:
                    theta_tilde_slice[k, l] = theta_tilde_val
                    stochastic_winner = choose_stochastic_winner(
                        theta_tilde_val, v_val, p, theta_points
                    )
                    theta_winner_slice[k, l] = stochastic_winner
        return theta_tilde_slice, theta_winner_slice

    tau_pairs = [(td, tf) for td in tau_d_grid for tf in tau_f_grid]

    if n_jobs == 1 or len(tau_pairs) <= 1:
        results_list = [
            compute_slice_for_taus(td, tf)
            for td, tf in tqdm(tau_pairs, desc="Processing tau pairs")
        ]
    else:
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(compute_slice_for_taus)(td, tf)
            for td, tf in tqdm(tau_pairs, desc="Processing tau pairs")
        )

    lookup_table_shape = (len(tau_d_grid), len(tau_f_grid), len(bar_grid), len(v_grid))
    theta_tilde_table = np.full(lookup_table_shape, np.nan)
    theta_winner_table = np.full(lookup_table_shape, np.nan)

    idx = 0
    for i, td in enumerate(tau_d_grid):
        for j, tf in enumerate(tau_f_grid):
            tilde_slice, winner_slice = results_list[idx]
            theta_tilde_table[i, j, :, :] = tilde_slice
            theta_winner_table[i, j, :, :] = winner_slice
            idx += 1

    # Helper index functions (closest grid index), as in the file
    def get_tau_d_index(val, grid=tau_d_grid):
        return int(np.argmin(np.abs(grid - val)))

    def get_tau_f_index(val, grid=tau_f_grid):
        return int(np.argmin(np.abs(grid - val)))

    def get_bar_beta_index(val, grid=bar_grid):
        return int(np.argmin(np.abs(grid - val)))

    def get_v_index(val, grid=v_grid):
        return int(np.argmin(np.abs(grid - val)))

    return (
        theta_tilde_table,
        theta_winner_table,
        get_tau_d_index,
        get_tau_f_index,
        get_bar_beta_index,
        get_v_index,
    )
