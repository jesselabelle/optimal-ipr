from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional

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
        self._profit_shifted = (self.tau_d > self.tau_f)
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
        if np.isclose(theta_tilde, 1.0):
            beta = np.array([min(1.0, float(bar_beta))])
        else:
            beta = self.beta_grid[self.beta_grid <= bar_beta]
            if beta.size == 0:
                return None

        # investor and imitator counts
        if np.isclose(theta_tilde, 1.0):
            num_investors = 1
        else:
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

        # innovator payoff before fees and own cost
        if self._profit_shifted:
            innovator_payoff = beta * (
                (1.0 - self.tau_f) * (v - self.c(theta_winner, v)) +
                (1.0 - self.tau_d) * self.c(theta_winner, v)
            )
        else:
            innovator_payoff = beta * v * (1.0 - self.tau_d)

        # innovator utility
        innov_util = innovator_payoff - self.Z(beta, v) - self.c(theta_winner, v)

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
            avg_imitator_cost = float(np.sum(self.c(imitator_thetas, v) * imitator_weights) / total_imit_mass)
        else:
            avg_imitator_cost = 0.0

        imitator_profit_pool = (1.0 - beta) * v
        profit_per_imitator = np.where(num_imitators > 0, imitator_profit_pool / num_imitators, 0.0)
        after_tax_profit_per_imit = profit_per_imitator * (1.0 - self.tau_d)
        avg_imitator_utility = after_tax_profit_per_imit - avg_imitator_cost
        imit_util_total = avg_imitator_utility * num_imitators

        # taxes and fees
        if self._profit_shifted:
            innov_taxable_base = self.c(theta_winner, v)
            foreign_tax_paid = self.tau_f * beta * (v - self.c(theta_winner, v))
        else:
            innov_taxable_base = v
            foreign_tax_paid = np.zeros_like(beta)

        tax_from_innov = self.tau_d * beta * innov_taxable_base
        tax_from_imits = self.tau_d * imitator_profit_pool if num_imitators > 0 else np.zeros_like(beta)
        tax_revenue = tax_from_innov + tax_from_imits
        public_revenue = tax_revenue + self.Z(beta, v)

        # conservation check: allocations sum to v
        innov_pre_cost_take_home = innovator_payoff - self.Z(beta, v)
        imits_pre_cost_take_home = after_tax_profit_per_imit * num_imitators
        total_value_distributed = (
            innov_pre_cost_take_home + imits_pre_cost_take_home + tax_revenue + foreign_tax_paid + self.Z(beta, v)
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
                welfare = np.zeros_like(beta) # Should not happen
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
        components = self._calculate_all_welfare_components(scheme, bar_beta, theta_tilde, theta_winner, v)
        if components is None:
            return None, None, None

        # participation constraints
        check = self.enforce_feasibility if enforce_feasibility is None else bool(enforce_feasibility)
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

    def debug_solve(self, scheme: str, bar_beta: float, theta_tilde: float, theta_winner: float, v: float) -> pd.DataFrame:
        comps = self._calculate_all_welfare_components(scheme, bar_beta, theta_tilde, theta_winner, v)
        if comps is None:
            return pd.DataFrame()
        return pd.DataFrame(comps)
