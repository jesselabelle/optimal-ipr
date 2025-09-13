import numpy as np

from optimal_ipr.regulator import RegulatorModel


def test_no_imitators_value_conservation():
    beta_grid = np.array([0.0, 0.5, 1.0])
    theta_points = np.array([0.5])
    theta_weights = np.array([1.0])

    def p(th, v):
        return np.ones_like(th, dtype=float)

    def c(th, v):
        return np.zeros_like(th, dtype=float)

    def Z(b, v):
        return np.zeros_like(b, dtype=float)

    def f_dist(th):
        return np.ones_like(th, dtype=float)

    def F_dist(t):
        return 0.0 if t < 1.0 else 1.0

    model = RegulatorModel(
        p_func=p,
        c_func=c,
        Z_func=Z,
        f_dist=f_dist,
        F_dist=F_dist,
        n_total_firms=1,
        tau_d=0.2,
        tau_f=0.1,
        preferences_dict={"utilitarian": "special_case"},
        beta_grid=beta_grid,
        theta_points=theta_points,
        theta_weights=theta_weights,
        enforce_feasibility=False,
    )

    comps = model._calculate_all_welfare_components(
        scheme="utilitarian",
        bar_beta=1.0,
        theta_tilde=0.0,
        theta_winner=0.5,
        v=1e-5,
    )

    assert comps is not None
    assert np.allclose(comps["beta"], beta_grid)
