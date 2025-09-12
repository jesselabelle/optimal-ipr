import numpy as np
import sys

sys.path.append("src")
from optimal_ipr.regulator.regulator_model import RegulatorModel


def test_welfare_components_match_baseline():
    p = lambda x, v: x
    c = lambda theta, v: 0.1 * v * np.ones_like(theta, dtype=float)
    Z = lambda beta, v: 0.05 * beta * v
    f_dist = lambda x: np.ones_like(x)
    F_dist = lambda x: x
    preferences = {
        "utilitarian": "special_case",
        "rawlsian": "special_case",
        "custom": {"phi": 0.5, "psi": 0.5},
    }
    beta_grid = np.linspace(0, 1, 5)
    theta_points = np.array([0.2, 0.4, 0.6, 0.8])
    theta_weights = np.array([0.25, 0.25, 0.25, 0.25])
    model = RegulatorModel(
        p,
        c,
        Z,
        f_dist,
        F_dist,
        n_total_firms=10,
        tau_d=0.1,
        tau_f=0.2,
        preferences_dict=preferences,
        beta_grid=beta_grid,
        theta_points=theta_points,
        theta_weights=theta_weights,
        enforce_feasibility=False,
    )
    res = model._calculate_all_welfare_components(
        "utilitarian", bar_beta=1.0, theta_tilde=0.5, theta_winner=0.7, v=100.0
    )
    expected_beta = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_innov_util = np.array([-10.0, 11.25, 32.5, 53.75, 75.0])
    expected_welfare = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
    assert np.allclose(res["beta"], expected_beta)
    assert np.allclose(res["innov_util"], expected_innov_util)
    assert np.allclose(res["welfare"], expected_welfare)
