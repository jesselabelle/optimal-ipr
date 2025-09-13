from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional
from optimal_ipr.regulator import RegulatorModel


class GovernmentModel:
    """
    Discrete government's problem, aligned with In[40]-In[41].
    Selects bar_beta by integrating over v with weights. Uses precomputed
    theta_tilde and theta_winner tables. Feasibility optional.
    """

    def __init__(
        self,
        *,
        tau_d: float,
        tau_f: float,
        gov_prefs: Dict[str, Callable[[np.ndarray], np.ndarray] | Callable[[float], float]],
        reg_prefs: Dict[str, Any],
        v_grid: np.ndarray,
        v_weights: np.ndarray,
        bar_grid: np.ndarray,
        master_lookup_table: np.ndarray,
        master_winner_table: np.ndarray,
        theta_points: np.ndarray,
        theta_weights: np.ndarray,
        enf_feas: bool,
        # dependencies to build RegulatorModel exactly as in the notebook
        p: Callable[[np.ndarray | float, float], np.ndarray | float],
        c: Callable[[np.ndarray | float, float], np.ndarray | float],
        Z: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
        f: Callable[[np.ndarray], np.ndarray],
        F: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
        n_total_firms: int,
        beta_grid: np.ndarray,
        # index helpers returned by build_lookup_tables
        get_tau_d_index: Callable[[float], int],
        get_tau_f_index: Callable[[float], int],
        get_bar_beta_index: Callable[[float], int],
        get_v_index: Callable[[float], int],
        # optional: agent type grid for the investor-count sanity check
        agent_types_grid: Optional[np.ndarray] = None,
    ):
        self.tau_d, self.tau_f = float(tau_d), float(tau_f)
        self.gov_prefs = gov_prefs
        self.reg_prefs = reg_prefs
        self.v_grid, self.v_weights = np.asarray(v_grid), np.asarray(v_weights)
        self.bar_grid = np.asarray(bar_grid)

        self.theta_points = np.asarray(theta_points)
        self.theta_weights = np.asarray(theta_weights)
        self.F = F
        self.N_total_firms = int(n_total_firms)
        self.enf_feas = bool(enf_feas)

        # slice the lookup tables at (tau_d, tau_f)
        idx_d = int(get_tau_d_index(self.tau_d))
        idx_f = int(get_tau_f_index(self.tau_f))
        self.theta_tilde_slice = master_lookup_table[idx_d, idx_f, :, :]
        self.theta_winner_slice = master_winner_table[idx_d, idx_f, :, :]

        # regulator sub-problem (same construction as notebook)
        self.regulator_solver = RegulatorModel(
            p_func=p,
            c_func=c,
            Z_func=Z,
            f_dist=f,
            F_dist=F,
            n_total_firms=n_total_firms,
            tau_d=self.tau_d,
            tau_f=self.tau_f,
            preferences_dict=reg_prefs,
            beta_grid=beta_grid,
            theta_points=self.theta_points,
            theta_weights=self.theta_weights,
            enforce_feasibility=self.enf_feas,
        )

        self._get_bar_beta_index = get_bar_beta_index
        self._get_v_index = get_v_index
        self._agent_types_grid = (
            np.asarray(agent_types_grid)
            if agent_types_grid is not None
            else np.linspace(0.0, 1.0, 500)
        )

    def _calculate_welfare_for_v(
        self,
        w: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
        regulator_scheme: str,
        bar_beta: float,
        v: float,
    ) -> Optional[Dict[str, float]]:
        idx_bar_beta = int(self._get_bar_beta_index(bar_beta))
        idx_v = int(self._get_v_index(v))
        theta_tilde = self.theta_tilde_slice[idx_bar_beta, idx_v]
        theta_winner = self.theta_winner_slice[idx_bar_beta, idx_v]

        if theta_tilde is None or np.isnan(theta_tilde) or (theta_tilde < 0) or (theta_tilde > 1):
            return None

        # sanity check as in notebook
        investor_types = self._agent_types_grid[self._agent_types_grid >= theta_tilde]
        if investor_types.size < 1:
            return None

        beta_star, tax_rev_star, reg_welfare_star = self.regulator_solver.solve(
            scheme=regulator_scheme,
            bar_beta=float(bar_beta),
            theta_tilde=float(theta_tilde),
            theta_winner=float(theta_winner),
            v=float(v),
            enforce_feasibility=self.enf_feas,
        )
        if beta_star is None:
            return None

        # full component table at the chosen (bar_beta, v)
        components_df = self.regulator_solver.debug_solve(
            regulator_scheme, float(bar_beta), float(theta_tilde), float(theta_winner), float(v)
        )
        if components_df.empty or not np.any(np.isclose(components_df["beta"], beta_star)):
            return None

        optimal_components = components_df[np.isclose(components_df["beta"], beta_star)].iloc[0]
        innov_util = float(optimal_components["innov_util"])
        imit_util = float(optimal_components["imit_util_total"])

        if self.enf_feas and ((innov_util <= 0.0) or (imit_util <= 0.0)):
            return None

        # transfers to non-investors
        F_theta_tilde = float(self.F(float(theta_tilde))) if theta_tilde < 1 else 1.0
        if theta_tilde >= 1:
            num_non_investors = self.N_total_firms - 1
        else:
            num_non_investors = int(F_theta_tilde * self.N_total_firms)
        transfer = float(tax_rev_star) / num_non_investors if num_non_investors > 0 else 0.0

        # average welfare weights
        non_investor_mask = self.theta_points < theta_tilde
        total_non_investor_prob_mass = float(np.sum(self.theta_weights[non_investor_mask]))
        avg_w_non_investor = 0.0
        if total_non_investor_prob_mass > 0:
            avg_w_non_investor = float(
                np.sum(
                    np.asarray(w(self.theta_points[non_investor_mask]))
                    * self.theta_weights[non_investor_mask]
                )
                / total_non_investor_prob_mass
            )

        total_welfare_non_investor = transfer * avg_w_non_investor * num_non_investors

        investor_mask = self.theta_points >= theta_tilde
        if theta_tilde >= 1:
            num_imitators = 0
        else:
            num_imitators = max(int((1.0 - F_theta_tilde) * self.N_total_firms) - 1, 0)
        total_investor_prob_mass = float(np.sum(self.theta_weights[investor_mask]))
        avg_w_imitator = 0.0
        if num_imitators > 0 and total_investor_prob_mass > 0:
            avg_w_imitator = float(
                np.sum(
                    np.asarray(w(self.theta_points[investor_mask]))
                    * self.theta_weights[investor_mask]
                )
                / total_investor_prob_mass
            )

        total_welfare_imitator = imit_util * avg_w_imitator

        total_welfare_innov = float(np.asarray(w(theta_winner))) * innov_util

        welfare_for_v = total_welfare_innov + total_welfare_imitator + total_welfare_non_investor

        return {
            "welfare_for_v": welfare_for_v,
            "theta_tilde": float(theta_tilde),
            "theta_winner": float(theta_winner),
            "beta_star": float(beta_star),
            "tax_revenue": float(tax_rev_star),
            "transfer": float(transfer),
            "total_welfare_innov": float(total_welfare_innov),
            "total_welfare_imitator": float(total_welfare_imitator),
            "total_welfare_non_investor": float(total_welfare_non_investor),
            "reg_welfare_for_v": float(reg_welfare_star),
        }

    def solve(self, *, government_scheme: str, regulator_scheme: str):
        """Return (optimal_row, full_results_df)."""
        w = self.gov_prefs[government_scheme]
        results_list = []
        for bar_beta in self.bar_grid:
            expected_welfare_gov = 0.0
            expected_beta_star = 0.0
            expected_reg_welfare = 0.0
            expected_innov_welfare = 0.0
            expected_imit_welfare = 0.0
            expected_non_invest_welfare = 0.0

            for v_idx, v in enumerate(self.v_grid):
                sim = self._calculate_welfare_for_v(w, regulator_scheme, float(bar_beta), float(v))
                if sim:
                    weight = float(self.v_weights[v_idx])
                    expected_welfare_gov += sim["welfare_for_v"] * weight
                    expected_beta_star += sim["beta_star"] * weight
                    expected_reg_welfare += sim["reg_welfare_for_v"] * weight
                    expected_innov_welfare += sim["total_welfare_innov"] * weight
                    expected_imit_welfare += sim["total_welfare_imitator"] * weight
                    expected_non_invest_welfare += sim["total_welfare_non_investor"] * weight

            results_list.append(
                {
                    "bar_beta": float(bar_beta),
                    "expected_welfare": float(expected_welfare_gov),
                    "expected_beta_star": float(expected_beta_star),
                    "expected_reg_welfare": float(expected_reg_welfare),
                    "expected_innov_welfare": float(expected_innov_welfare),
                    "expected_imit_welfare": float(expected_imit_welfare),
                    "expected_non_invest_welfare": float(expected_non_invest_welfare),
                }
            )

        if not results_list:
            return None, None
        results_df = pd.DataFrame(results_list)
        optimal_row = results_df.loc[results_df["expected_welfare"].idxmax()]
        return optimal_row, results_df

    def debug_solve(self, *, government_scheme: str, regulator_scheme: str) -> pd.DataFrame:
        """Return full bar_beta-by-components table."""
        w = self.gov_prefs[government_scheme]
        debug_results = []
        for bar_beta in self.bar_grid:
            expected_welfare_gov = 0.0
            expected_beta_star = 0.0
            expected_reg_welfare = 0.0
            expected_innov_welfare = 0.0
            expected_imit_welfare = 0.0
            expected_non_invest_welfare = 0.0

            for v_idx, v in enumerate(self.v_grid):
                sim = self._calculate_welfare_for_v(w, regulator_scheme, float(bar_beta), float(v))
                if sim:
                    weight = float(self.v_weights[v_idx])
                    expected_welfare_gov += sim["welfare_for_v"] * weight
                    expected_beta_star += sim["beta_star"] * weight
                    expected_reg_welfare += sim["reg_welfare_for_v"] * weight
                    expected_innov_welfare += sim["total_welfare_innov"] * weight
                    expected_imit_welfare += sim["total_welfare_imitator"] * weight
                    expected_non_invest_welfare += sim["total_welfare_non_investor"] * weight

            debug_results.append(
                {
                    "bar_beta": float(bar_beta),
                    "expected_welfare": float(expected_welfare_gov),
                    "expected_beta_star": float(expected_beta_star),
                    "expected_reg_welfare": float(expected_reg_welfare),
                    "expected_innov_welfare": float(expected_innov_welfare),
                    "expected_imit_welfare": float(expected_imit_welfare),
                    "expected_non_invest_welfare": float(expected_non_invest_welfare),
                }
            )
        return pd.DataFrame(debug_results)
