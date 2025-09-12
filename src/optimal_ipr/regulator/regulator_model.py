from __future__ import annotations

import numpy as np
from numba import njit
from typing import Callable, Dict, Any, Optional


@njit
def _welfare_components_loop(
    beta: np.ndarray,
    profit_shifted: bool,
    tau_d: float,
    tau_f: float,
    v: float,
    innov_cost: float,
    Z_beta: np.ndarray,
    num_imitators: int,
    avg_imit_cost: float,
):
    n = beta.shape[0]
    innov_util = np.empty(n, dtype=np.float64)
    imit_util_total = np.empty(n, dtype=np.float64)
    tax_revenue = np.empty(n, dtype=np.float64)
    public_revenue = np.empty(n, dtype=np.float64)
    innov_pre_cost_take_home = np.empty(n, dtype=np.float64)
    after_tax_profit_per_imit = np.empty(n, dtype=np.float64)
    foreign_tax_paid = np.empty(n, dtype=np.float64)

    for i in range(n):
        b = beta[i]
        if profit_shifted:
            innovator_payoff = b * ((1.0 - tau_f) * (v - innov_cost) + (1.0 - tau_d) * innov_cost)
            innov_taxable_base = innov_cost
            foreign_tax_paid[i] = tau_f * b * (v - innov_cost)
        else:
            innovator_payoff = b * v * (1.0 - tau_d)
            innov_taxable_base = v
            foreign_tax_paid[i] = 0.0

        innov_util[i] = innovator_payoff - Z_beta[i] - innov_cost

        imitator_profit_pool = (1.0 - b) * v
        if num_imitators > 0:
            profit_per_imitator = imitator_profit_pool / num_imitators
            after_tax = profit_per_imitator * (1.0 - tau_d)
            after_tax_profit_per_imit[i] = after_tax
            avg_imit_utility = after_tax - avg_imit_cost
            imit_util_total[i] = avg_imit_utility * num_imitators
            tax_from_imits = tau_d * imitator_profit_pool
        else:
            after_tax_profit_per_imit[i] = 0.0
            imit_util_total[i] = 0.0
            tax_from_imits = 0.0

        tax_from_innov = tau_d * b * innov_taxable_base
        tax_revenue[i] = tax_from_innov + tax_from_imits
        public_revenue[i] = tax_revenue[i] + Z_beta[i]
        innov_pre_cost_take_home[i] = innovator_payoff - Z_beta[i]

    return (
        innov_util,
        imit_util_total,
        tax_revenue,
        public_revenue,
        innov_pre_cost_take_home,
        after_tax_profit_per_imit,
        foreign_tax_paid,
    )


class RegulatorModel:
    """
    Discrete regulator problem matching In[18].
    Builds welfare over feasible beta <= bar_beta and selects beta_star.
    """

    def __init__(
        self,
        p_func: Callable[[np.ndarray | float, float], np.ndarray | float],
        c_func: Callable[[np.ndarray | float, float], np.ndarray | float],
        Z_func: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
        f_dist: Callable[[np.ndarray], np.ndarray],
        F_dist: Callable[[float], float],
        n_total_firms: int,
        tau_d: float,
        tau_f: float,
        preferences_dict: Dict[str, Any],
        beta_grid: np.ndarray,
        theta_points: np.ndarray,
        theta_weights: np.ndarray,
        enforce_feasibility: bool,
    ):
        # functions and primitives
        self.p = p_func
        self.c = c_func
        self.Z = Z_func
        self.f = f_dist
        self.F = F_dist
        self.n_total_firms = float(n_total_firms)
        self.tau_d = float(tau_d)
        self.tau_f = float(tau_f)
        self.preferences = preferences_dict
        self._profit_shifted = self.tau_d > self.tau_f
        self.beta_grid = np.asarray(beta_grid, dtype=float)
        self.theta_points = np.asarray(theta_points, dtype=float)
        self.theta_weights = np.asarray(theta_weights, dtype=float)
        self.enforce_feasibility = bool(enforce_feasibility)

    def _calculate_all_welfare_components(
        self,
        scheme: str,
        bar_beta: float,
        theta_tilde: float,
        theta_winner: float,
        v: float,
    ) -> Optional[Dict[str, np.ndarray]]:
        # feasible betas
        beta = self.beta_grid[self.beta_grid <= bar_beta]
        if beta.size == 0:
            return None

        # investor and imitator counts
        num_investors = int((1.0 - self.F(theta_tilde)) * self.n_total_firms)
        if num_investors < 1:
            return {
                "beta": beta,
                "innov_util": np.zeros_like(beta),
                "imit_util_total": np.zeros_like(beta),
                "tax_revenue": np.zeros_like(beta),
                "public_revenue": np.zeros_like(beta),
                "welfare": np.zeros_like(beta),
            }
        num_imitators = max(num_investors - 1, 0)

        innov_cost = float(self.c(theta_winner, v))

        # imitators: exclude the winner, weight by theta_weights
        investor_mask = self.theta_points >= theta_tilde
        winner_idx = int(np.argmin(np.abs(self.theta_points - theta_winner)))
        imitator_mask = investor_mask.copy()
        if 0 <= winner_idx < imitator_mask.size:
            imitator_mask[winner_idx] = False

        imitator_thetas = self.theta_points[imitator_mask]
        imitator_weights = self.theta_weights[imitator_mask]
        total_imit_mass = float(np.sum(imitator_weights))

        if total_imit_mass > 0:
            avg_imitator_cost = float(
                np.sum(self.c(imitator_thetas, v) * imitator_weights) / total_imit_mass
            )
        else:
            avg_imitator_cost = 0.0

        Z_beta = self.Z(beta, v)
        (
            innov_util,
            imit_util_total,
            tax_revenue,
            public_revenue,
            innov_pre_cost_take_home,
            after_tax_profit_per_imit,
            foreign_tax_paid,
        ) = _welfare_components_loop(
            beta,
            self._profit_shifted,
            self.tau_d,
            self.tau_f,
            v,
            innov_cost,
            Z_beta,
            num_imitators,
            avg_imitator_cost,
        )

        avg_imitator_utility = after_tax_profit_per_imit - avg_imitator_cost

        # conservation check: allocations sum to v
        imits_pre_cost_take_home = after_tax_profit_per_imit * num_imitators
        total_value_distributed = (
            innov_pre_cost_take_home
            + imits_pre_cost_take_home
            + tax_revenue
            + foreign_tax_paid
            + Z_beta
        )
        if not np.all(np.isclose(total_value_distributed, v)):
            raise ValueError(
                "Value conservation error: distributed total differs from v. "
                f"v={v}, total={total_value_distributed}"
            )

        # Calculate Total Welfare based on the regulator's preference scheme.
        scheme_params = self.preferences[scheme]
        if scheme_params == "special_case":
            if scheme == "utilitarian":
                # Sum of all utilities (innovator, imitators, public revenue)
                welfare = innov_util + imit_util_total + public_revenue
            elif scheme == "rawlsian":
                # Welfare is the utility of the worst-off agent (min of innovator vs avg imitator)
                welfare = np.minimum(innov_util, avg_imitator_utility)
            else:
                welfare = np.zeros_like(beta)  # Should not happen
        else:
            # Weighted sum based on phi (private vs. private) and psi (private vs. public)
            phi, psi = scheme_params["phi"], scheme_params["psi"]
            private_welfare = phi * innov_util + (1 - phi) * imit_util_total
            welfare = private_welfare + psi * public_revenue

        return {
            "beta": beta,
            "innov_util": innov_util,
            "imit_util_total": imit_util_total,
            "tax_revenue": tax_revenue,
            "public_revenue": public_revenue,
            "welfare": welfare,
        }

    def solve(
        self,
        scheme: str,
        bar_beta: float,
        theta_tilde: float,
        theta_winner: float,
        v: float,
        enforce_feasibility: Optional[bool] = None,
    ):
        """Return (beta_star, tax_revenue_star, reg_welfare_star)."""
        components = self._calculate_all_welfare_components(
            scheme, bar_beta, theta_tilde, theta_winner, v
        )
        if components is None:
            return None, None, None

        # participation constraints
        check = (
            self.enforce_feasibility if enforce_feasibility is None else bool(enforce_feasibility)
        )
        if check:
            feasible = (components["innov_util"] > 0) & (components["imit_util_total"] > 0)
            if not np.any(feasible):
                return None, None, None
            welfare = np.where(feasible, components["welfare"], -np.inf)
            idx = int(np.argmax(welfare))
        else:
            idx = int(np.argmax(components["welfare"]))

        beta_star = float(components["beta"][idx])
        tax_revenue_star = float(components["tax_revenue"][idx])
        reg_welfare_star = float(components["welfare"][idx])
        return beta_star, tax_revenue_star, reg_welfare_star

    def debug_solve(
        self, scheme: str, bar_beta: float, theta_tilde: float, theta_winner: float, v: float
    ):
        comps = self._calculate_all_welfare_components(
            scheme, bar_beta, theta_tilde, theta_winner, v
        )
        if comps is None:
            return {}
        try:
            import pandas as pd  # type: ignore

            return pd.DataFrame(comps)
        except Exception:
            return comps
