import numpy as np
import pandas as pd
from optimal_ipr.baseline.baseline_model import BaselineModel
from optimal_ipr.government.government_model import GovernmentModel


class DummyBaseline(BaselineModel):
    def __init__(self):
        self.gov_prefs = {"linear": lambda x: x}
        self.v_grid = np.array([1.0, 2.0, 3.0])
        self.v_weights = np.array([0.2, 0.3, 0.5])

    def _calculate_components_for_v(self, w, v):
        return {"gov_welfare": v * 10, "innov_util": v * 2, "non_innov_util": v}


def test_baseline_model_vectorization():
    model = DummyBaseline()
    result = model.solve("linear")
    expected = pd.Series(
        {
            "gov_welfare": np.dot([10, 20, 30], model.v_weights),
            "innov_util": np.dot([2, 4, 6], model.v_weights),
            "non_innov_util": np.dot([1, 2, 3], model.v_weights),
        },
        name="baseline_results",
    )
    pd.testing.assert_series_equal(result, expected)


class DummyGovernment(GovernmentModel):
    def __init__(self):
        self.gov_prefs = {"const": lambda x: x}
        self.bar_grid = np.array([0.0, 1.0])
        self.v_grid = np.array([1.0, 2.0])
        self.v_weights = np.array([0.4, 0.6])

    def _calculate_welfare_for_v(self, w, regulator_scheme, bar_beta, v):
        return {
            "welfare_for_v": v + bar_beta,
            "beta_star": v * bar_beta,
            "reg_welfare_for_v": v - bar_beta,
            "innov_util": 2 * v,
            "imit_util": v,
            "non_invest_util": 0.5 * v,
        }


def test_government_model_vectorization():
    model = DummyGovernment()
    opt_row, results_df = model.solve(government_scheme="const", regulator_scheme="dummy")
    expected = pd.DataFrame(
        [
            {
                "bar_beta": 0.0,
                "expected_welfare": np.dot(model.v_weights, [1.0, 2.0]),
                "expected_beta_star": 0.0,
                "expected_reg_welfare": np.dot(model.v_weights, [1.0, 2.0]),
                "expected_innov_util": np.dot(model.v_weights, [2.0, 4.0]),
                "expected_non_innov_util": np.dot(model.v_weights, [1.5, 3.0]),
            },
            {
                "bar_beta": 1.0,
                "expected_welfare": np.dot(model.v_weights, [2.0, 3.0]),
                "expected_beta_star": np.dot(model.v_weights, [1.0, 2.0]),
                "expected_reg_welfare": np.dot(model.v_weights, [0.0, 1.0]),
                "expected_innov_util": np.dot(model.v_weights, [2.0, 4.0]),
                "expected_non_innov_util": np.dot(model.v_weights, [1.5, 3.0]),
            },
        ]
    )
    pd.testing.assert_frame_equal(results_df, expected)
    assert opt_row["bar_beta"] == 1.0
    assert opt_row["expected_welfare"] == expected.loc[1, "expected_welfare"]
