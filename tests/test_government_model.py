import numpy as np
import pandas as pd

from optimal_ipr.government import GovernmentModel


class DummyRegulator:
    def solve(self, scheme, bar_beta, theta_tilde, theta_winner, v, enforce_feasibility):
        return 1.0, 1.0, 1.0

    def debug_solve(self, scheme, bar_beta, theta_tilde, theta_winner, v):
        return pd.DataFrame(
            {
                "beta": [1.0],
                "innov_util": [1.0],
                "imit_util_total": [0.0],
                "tax_revenue": [1.0],
                "public_revenue": [1.0],
                "welfare": [1.0],
            }
        )


def create_government_model(theta_tilde_value: float) -> GovernmentModel:
    v_grid = np.array([1.0])
    v_weights = np.array([1.0])
    bar_grid = np.array([1.0])
    master_lookup = np.array([[[[theta_tilde_value]]]])
    master_winner = np.array([[[[theta_tilde_value]]]])
    theta_points = np.array([0.0, 1.0])
    theta_weights = np.array([0.5, 0.5])

    def F(t):
        return float(np.clip(t, 0.0, 1.0))

    gm = GovernmentModel(
        tau_d=0.2,
        tau_f=0.1,
        gov_prefs={},
        reg_prefs={"utilitarian": {"phi": 1.0, "psi": 1.0}},
        v_grid=v_grid,
        v_weights=v_weights,
        bar_grid=bar_grid,
        master_lookup_table=master_lookup,
        master_winner_table=master_winner,
        theta_points=theta_points,
        theta_weights=theta_weights,
        enf_feas=False,
        p=lambda th, v: np.ones_like(th, dtype=float),
        c=lambda th, v: np.zeros_like(th, dtype=float),
        Z=lambda b, v: np.zeros_like(b, dtype=float),
        f=lambda t: np.ones_like(t, dtype=float),
        F=F,
        n_total_firms=2,
        beta_grid=np.array([1.0]),
        get_tau_d_index=lambda x: 0,
        get_tau_f_index=lambda x: 0,
        get_bar_beta_index=lambda x: 0,
        get_v_index=lambda x: 0,
        agent_types_grid=np.array([0.0, 1.0]),
    )
    gm.regulator_solver = DummyRegulator()
    return gm


def test_theta_tilde_boundaries():
    gm = create_government_model(1.0)
    res = gm._calculate_welfare_for_v(
        w=lambda t: np.ones_like(t, dtype=float),
        regulator_scheme="utilitarian",
        bar_beta=1.0,
        v=1.0,
    )
    assert res is not None
    assert np.isclose(res["theta_tilde"], 1.0)
    assert np.isclose(res["total_welfare_imitator"], 0.0)

    gm.theta_tilde_slice[0, 0] = 0.0
    gm.theta_winner_slice[0, 0] = 0.0
    res0 = gm._calculate_welfare_for_v(
        w=lambda t: np.ones_like(t, dtype=float),
        regulator_scheme="utilitarian",
        bar_beta=1.0,
        v=1.0,
    )
    assert res0 is not None
    assert np.isclose(res0["theta_tilde"], 0.0)

    gm.theta_tilde_slice[0, 0] = 1.1
    res_invalid = gm._calculate_welfare_for_v(
        w=lambda t: np.ones_like(t, dtype=float),
        regulator_scheme="utilitarian",
        bar_beta=1.0,
        v=1.0,
    )
    assert res_invalid is None
