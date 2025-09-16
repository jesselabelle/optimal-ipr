from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Sequence
from optimal_ipr.government import GovernmentModel
from optimal_ipr.baseline import BaselineModel

_N_TOTAL_FIRMS = 29_990

def _to_grid(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))

def welfare_outcomes(
    *,
    tau_d: Sequence[float] | float,
    tau_f: Sequence[float] | float,
    gov_prefs: Dict[str, Callable[[np.ndarray | float], np.ndarray | float]],
    reg_prefs: Dict[str, Any],
    v_grid: np.ndarray,
    v_weights: np.ndarray,
    theta_tilde_table: np.ndarray,
    theta_winner_table: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    F: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
    F_inv: Callable[[np.ndarray], np.ndarray],  # accepted for parity; not used here
    p: Callable[[np.ndarray | float, float], np.ndarray | float],
    c: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
    Z: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
    feas: bool,
    get_tau_d_index: Callable[[float], int] | None = None,
    get_tau_f_index: Callable[[float], int] | None = None,
    get_bar_beta_index: Callable[[float], int] | None = None,
    get_v_index: Callable[[float], int] | None = None,
) -> pd.DataFrame:
    """
    In[25]: Solve the gov problem for each (tau_d, tau_f), compute baseline (beta=1),
    then return the table with % changes relative to baseline.
    """
    tau_d_grid = _to_grid(tau_d)
    tau_f_grid = _to_grid(tau_f)

    # use lookup-table indexers when provided; otherwise infer from the caller grids
    if get_tau_d_index is None:
        lookup_tau_d_len = int(theta_tilde_table.shape[0])
        if tau_d_grid.shape[0] != lookup_tau_d_len:
            raise ValueError(
                "tau_d grid length does not match lookup table axis. "
                "Pass the indexer returned by build_lookup_tables to evaluate a subset of tau_d values."
            )

        def local_get_tau_d_index(val, grid=tau_d_grid):
            return int(np.argmin(np.abs(grid - val)))
    else:
        local_get_tau_d_index = get_tau_d_index

    if get_tau_f_index is None:
        lookup_tau_f_len = int(theta_tilde_table.shape[1])
        if tau_f_grid.shape[0] != lookup_tau_f_len:
            raise ValueError(
                "tau_f grid length does not match lookup table axis. "
                "Pass the indexer returned by build_lookup_tables to evaluate a subset of tau_f values."
            )

        def local_get_tau_f_index(val, grid=tau_f_grid):
            return int(np.argmin(np.abs(grid - val)))
    else:
        local_get_tau_f_index = get_tau_f_index

    # bar_beta grid inferred from lookup tables
    n_beta = int(theta_tilde_table.shape[2])
    bar_grid = np.linspace(0.0, 1.0, n_beta)
    beta_grid = bar_grid.copy()

    # theta integration grid
    theta_points = np.linspace(0.0, 1.0, 10000)
    theta_weights = f(theta_points)

    # index helpers aligned with lookup axes
    if get_bar_beta_index is None:
        def local_get_bar_beta_index(val, grid=bar_grid):
            return int(np.argmin(np.abs(grid - val)))
    else:
        local_get_bar_beta_index = get_bar_beta_index

    if get_v_index is None:
        def local_get_v_index(val, grid=v_grid):
            return int(np.argmin(np.abs(grid - val)))
    else:
        local_get_v_index = get_v_index

    # 1) Main simulations across schemes
    records = []
    for td in tau_d_grid:
        for tf in tau_f_grid:
            gov_model = GovernmentModel(
                tau_d=float(td), tau_f=float(tf),
                gov_prefs=gov_prefs, reg_prefs=reg_prefs,
                v_grid=v_grid, v_weights=v_weights, bar_grid=bar_grid,
                master_lookup_table=theta_tilde_table, master_winner_table=theta_winner_table,
                theta_points=theta_points, theta_weights=theta_weights,
                enf_feas=feas,
                p=p, c=c, Z=Z, f=f, F=lambda t: float(F(np.array([t]))) if callable(F) else F,
                n_total_firms=_N_TOTAL_FIRMS, beta_grid=beta_grid,
                get_tau_d_index=local_get_tau_d_index, get_tau_f_index=local_get_tau_f_index,
                get_bar_beta_index=local_get_bar_beta_index, get_v_index=local_get_v_index,
                agent_types_grid=theta_points,
            )
            for gov_name in gov_prefs.keys():
                for reg_name in reg_prefs.keys():
                    optimal_row, _ = gov_model.solve(government_scheme=gov_name, regulator_scheme=reg_name)
                    if optimal_row is None:
                        continue
                    records.append({
                        "tau_d": float(td),
                        "tau_f": float(tf),
                        "gov_scheme": gov_name,
                        "reg_scheme": reg_name,
                        "bar_beta": float(optimal_row["bar_beta"]),
                        "beta_star": float(optimal_row["expected_beta_star"]),
                        "expected_welfare": float(optimal_row["expected_welfare"]),
                        "expected_innov_welfare": float(optimal_row["expected_innov_welfare"]),
                        "expected_imit_welfare": float(optimal_row["expected_imit_welfare"]),
                        "expected_non_invest_welfare": float(optimal_row["expected_non_invest_welfare"]),
                    })
    main_df = pd.DataFrame.from_records(records)
    if main_df.empty:
        return pd.DataFrame(columns=[
            "Tau D","Tau F","Gov Pref","Reg Pref","Optimal Patent Breadth Cap", "Expected Optimal Patent Breadth Granted",
            "Welfare % Change","Innovator Welfare % Change", "Imitator Welfare % Change", 
            "Non-Investor Welfare % Change",
        ])

    # 2) Baseline per (tau_d, tau_f, gov_scheme)
    base_records = []
    for td in tau_d_grid:
        for tf in tau_f_grid:
            baseline = BaselineModel(
                tau_d=float(td), tau_f=float(tf), gov_prefs=gov_prefs,
                v_grid=v_grid, v_weights=v_weights,
                master_lookup_table=theta_tilde_table, master_winner_table=theta_winner_table,
                theta_points=theta_points, theta_weights=theta_weights,
                F=lambda t: float(F(np.array([t]))) if callable(F) else F,
                c=c, Z=Z, n_total_firms=_N_TOTAL_FIRMS,
                get_tau_d_index=local_get_tau_d_index, get_tau_f_index=local_get_tau_f_index,
                get_bar_beta_index=local_get_bar_beta_index, get_v_index=local_get_v_index,
            )
            for gov_name in gov_prefs.keys():
                base_row = baseline.solve(government_scheme=gov_name)
                base_records.append({
                    "tau_d": float(td),
                    "tau_f": float(tf),
                    "gov_scheme": gov_name,
                    "baseline_gov_welfare": float(base_row["gov_welfare"]),
                    "baseline_innov_welfare": float(base_row["total_welfare_innov"]),
                    "baseline_imit_welfare": float(base_row["total_welfare_imitator"]),
                    "baseline_non_invest_welfare": float(base_row["total_welfare_non_investor"]),
                })
    base_df = pd.DataFrame.from_records(base_records)

    # 3) Merge and compute percent changes
    results_df = main_df.merge(base_df, on=["tau_d","tau_f","gov_scheme"], how="left")

    def pct_change(actual, baseline):
        if pd.isna(baseline) or pd.isna(actual) or baseline == 0:
            return np.nan
        return 100.0 * (actual - baseline) / abs(baseline)

    results_df["welfare_pct_change"] = results_df.apply(
        lambda r: pct_change(r["expected_welfare"], r["baseline_gov_welfare"]), axis=1
    )
    results_df["innov_welfare_pct_change"] = results_df.apply(
        lambda r: pct_change(r["expected_innov_welfare"], r["baseline_innov_welfare"]), axis=1
    )
    results_df["imit_welfare_pct_change"] = results_df.apply(
        lambda r: pct_change(r["expected_imit_welfare"], r["baseline_imit_welfare"]), axis=1
    )
    results_df["non_invest_welfare_pct_change"] = results_df.apply(
        lambda r: pct_change(r["expected_non_invest_welfare"], r["baseline_non_invest_welfare"]), axis=1
    )

    # 4) Final table
    final_cols = {
        "tau_d": "Tau D",
        "tau_f": "Tau F",
        "gov_scheme": "Gov Pref",
        "reg_scheme": "Reg Pref",
        "bar_beta": "Optimal Patent Breadth Cap",
        "beta_star": "Expected Optimal Patent Breadth Granted",
        "welfare_pct_change": "Welfare % Change",
        "innov_welfare_pct_change": "Innovator Welfare % Change",
        "imit_welfare_pct_change": "Imitator Welfare % Change",
        "non_invest_welfare_pct_change": "Non-Investor Welfare % Change",
    }
    return results_df[list(final_cols.keys())].rename(columns=final_cols)