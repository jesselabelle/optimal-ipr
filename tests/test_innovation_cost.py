import numpy as np
from optimal_ipr.cost import build_cost_function

def _uniform_pdf(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

def test_calibration_and_shape():
    # With uniform f on [0,1], E[(1-theta)^gamma] = 1/(gamma+1)
    TARGET = 0.30
    GAMMA = 2.0  # E = 1/3
    MIN_COST = 44_425 / 1e6
    c = build_cost_function(
        _uniform_pdf,
        TARGET,
        GAMMA,
    )

    # Check average cost share equals TARGET when v=1 and theta~uniform
    grid = np.linspace(0, 1, 10001)
    shares = c(grid, 1.0)
    avg_share = np.trapezoid(shares, grid) / (grid[-1] - grid[0])
    assert np.isclose(avg_share, TARGET, atol=5e-4)

    # Minimum cost component is constant and independent of theta and v
    base_cost = c(grid, 0.0)
    assert np.allclose(base_cost, MIN_COST)

    # Only the variable component should scale with v
    variable_component = c(0.25, 1.0) - MIN_COST
    assert np.isclose(c(0.25, 3.0) - MIN_COST, 3.0 * variable_component)

    # Within each region the cost is decreasing in theta for fixed v
    v = 2.0
    s = c(grid, v)
    assert np.all(np.diff(s) <= 1e-10)
