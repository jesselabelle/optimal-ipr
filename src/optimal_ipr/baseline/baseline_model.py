from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any


class BaselineModel:
    """
    Baseline scenario with beta fixed at 1.
    Pulls θ̃, θ_winner from lookup at bar_beta=1 and computes outcomes.
    """

    def __init__(
        self,
        *,
        tau_d: float,
        tau_f: float,
        gov_prefs: Dict[str, Callable[[np.ndarray | float], np.ndarray | float]],
        v_grid: np.ndarray,
        v_weights: np.ndarray,
        master_lookup_table: np.ndarray,
        master_winner_table: np.ndarray,
        theta_points: np.ndarray,
        theta_weights: np.ndarray,
        F: Callable[[float], float],
        c: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
        Z: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
        n_total_firms: int,
        get_tau_d_index: Callable[[float], int],
        get_tau_f_index: Callable[[float], int],
        get_bar_beta_index: Callable[[float], int],
        get_v_index: Callable[[float], int],
    ):
        self.tau_d = float(tau_d)
        self.tau_f = float(tau_f)
        self.gov_prefs = gov_prefs
        self.v_grid = np.asarray(v_grid, dtype=float)
        self.v_weights = np.asarray(v_weights, dtype=float)
        self.theta_points = np.asarray(theta_points, dtype=float)
        self.theta_weights = np.asarray(theta_weights, dtype=float)
        self.F, self.c, self.Z = F, c, Z
        self.N_total_firms = int(n_total_firms)
        self._profit_shifted = self.tau_d > self.tau_f

        # bar_beta = 1 lookup slices (exactly as notebook does). :contentReference[oaicite:2]{index=2}
        idx_d = int(get_tau_d_index(self.tau_d))
        idx_f = int(get_tau_f_index(self.tau_f))
        idx_bar_beta_one = int(get_bar_beta_index(1.0))
        self.theta_tilde_for_v = master_lookup_table[idx_d, idx_f, idx_bar_beta_one, :]
        self.theta_winner_for_v = master_winner_table[idx_d, idx_f, idx_bar_beta_one, :]

        # store indexers
        self._get_v_index = get_v_index

    def _calculate_components_for_v(
        self, w: Callable[[np.ndarray | float], np.ndarray | float], v: float
    ):
        """
        Compute all pieces for a single v at beta=1. Mirrors notebook In[23].
        """
        # Pull θ̃, θ_winner for this v. :contentReference[oaicite:3]{index=3}
        idx_v = int(self._get_v_index(v))
        theta_tilde = float(self.theta_tilde_for_v[idx_v])
        theta_winner = float(self.theta_winner_for_v[idx_v])
        if np.isnan(theta_tilde) or np.isnan(theta_winner):
            return None  # no investment case, skip. :contentReference[oaicite:4]{index=4}

        beta = 1.0  # fixed baseline. :contentReference[oaicite:5]{index=5}

        # Innovator payoff and utility. :contentReference[oaicite:6]{index=6}
        if self._profit_shifted:
            innovator_payoff = beta * (
                (1 - self.tau_f) * (v - self.c(theta_winner, v))
                + (1 - self.tau_d) * self.c(theta_winner, v)
            )
        else:
            innovator_payoff = beta * v * (1 - self.tau_d)
        innov_util = innovator_payoff - self.Z(beta, v) - self.c(theta_winner, v)

        # Imitator utility: zero revenue, pay average cost; exclude winner from investor set. :contentReference[oaicite:7]{index=7}
        num_investors = int((1 - self.F(theta_tilde)) * self.N_total_firms)
        num_imitators = num_investors - 1
        investor_mask = self.theta_points >= theta_tilde
        winner_idx_in_grid = int(np.argmin(np.abs(self.theta_points - theta_winner)))
        imitator_mask = investor_mask.copy()
        if 0 <= winner_idx_in_grid < imitator_mask.size:
            imitator_mask[winner_idx_in_grid] = False
        imitator_thetas = self.theta_points[imitator_mask]
        imitator_weights = self.theta_weights[imitator_mask]
        avg_imitator_cost = 0.0
        if np.sum(imitator_weights) > 0:
            avg_imitator_cost = float(
                np.sum(self.c(imitator_thetas, v) * imitator_weights) / np.sum(imitator_weights)
            )
        imit_util_total = -avg_imitator_cost * num_imitators if num_imitators > 0 else 0.0

        # Public revenue components; no imitator taxes; add fees. :contentReference[oaicite:8]{index=8}
        if self._profit_shifted:
            innov_taxable_base = self.c(theta_winner, v)
            foreign_tax_paid = self.tau_f * beta * (v - self.c(theta_winner, v))
        else:
            innov_taxable_base = v
            foreign_tax_paid = 0.0
        tax_from_innov = self.tau_d * beta * innov_taxable_base
        tax_revenue = tax_from_innov
        public_revenue = tax_revenue + self.Z(beta, v)

        # Value conservation check. Sum of allocations equals v. :contentReference[oaicite:9]{index=9}
        total_value_distributed = (
            (innovator_payoff - self.Z(beta, v)) + public_revenue + foreign_tax_paid
        )
        if not np.isclose(total_value_distributed, v):
            raise ValueError("Value conservation error in BaselineModel.")

        # Transfers to non-investors from tax revenue. :contentReference[oaicite:10]{index=10}
        num_non_investors = int(self.F(theta_tilde) * self.N_total_firms)
        transfer = tax_revenue / num_non_investors if num_non_investors > 0 else 0.0

        # Government welfare aggregation. Non-investors and imitators weighted by w(θ). :contentReference[oaicite:11]{index=11}
        non_investor_mask = self.theta_points < theta_tilde
        avg_w_non_investor = 0.0
        denom_non = float(np.sum(self.theta_weights[non_investor_mask]))
        if denom_non > 0:
            avg_w_non_investor = float(
                np.sum(
                    np.asarray(w(self.theta_points[non_investor_mask]))
                    * self.theta_weights[non_investor_mask]
                )
                / denom_non
            )
        total_welfare_non_investor = transfer * avg_w_non_investor * num_non_investors

        non_invest_util = transfer * num_non_investors
        total_non_innov_util = non_invest_util + imit_util_total

        # Imitator welfare weight over imitators only. :contentReference[oaicite:12]{index=12}
        avg_w_imitator = 0.0
        denom_inv = float(np.sum(imitator_weights))
        if num_imitators > 0 and denom_inv > 0:
            avg_w_imitator = float(
                np.sum(np.asarray(w(imitator_thetas)) * imitator_weights) / denom_inv
            )
        total_welfare_imitator = imit_util_total * avg_w_imitator

        # Government welfare for this v. :contentReference[oaicite:13]{index=13}
        gov_welfare = (
            float(np.asarray(w(theta_winner))) * innov_util
            + total_welfare_imitator
            + total_welfare_non_investor
        )

        return {
            "gov_welfare": float(gov_welfare),
            "innov_util": float(innov_util),
            "non_innov_util": float(total_non_innov_util),
        }

    def solve(self, government_scheme: str) -> pd.Series:
        """
        Expected baseline outcomes by averaging over v with weights. :contentReference[oaicite:14]{index=14}
        """
        w = self.gov_prefs[government_scheme]
        components = [self._calculate_components_for_v(w, float(v)) for v in self.v_grid]
        gov_welfare_arr = np.array(
            [comp["gov_welfare"] if comp else 0.0 for comp in components], dtype=float
        )
        innov_util_arr = np.array(
            [comp["innov_util"] if comp else 0.0 for comp in components], dtype=float
        )
        non_innov_util_arr = np.array(
            [comp["non_innov_util"] if comp else 0.0 for comp in components], dtype=float
        )
        acc = {
            "gov_welfare": float(np.dot(gov_welfare_arr, self.v_weights)),
            "innov_util": float(np.dot(innov_util_arr, self.v_weights)),
            "non_innov_util": float(np.dot(non_innov_util_arr, self.v_weights)),
        }
        return pd.Series(acc, name="baseline_results")
