import numpy as np
from optimal_ipr.distributions import value_distribution

def test_weights_sum_to_one():
    v_grid, v_weight = value_distribution(51, 3.0)
    assert np.isclose(v_weight.sum(), 1.0)
    assert len(v_grid) == len(v_weight) == 51