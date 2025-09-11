import numpy as np
from optimal_ipr.cost import build_cost_function

def _uniform_pdf(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

def test_calibration_and_shape():
    # With uniform f on [0,1], E[(1-theta)^gamma] = 1/(gamma+1)
    TARGET = 0.30
    C_MIN = 0.10
    GAMMA = 2.0  # E = 1/3
    c = build_cost_function(_uniform_pdf, TARGET, C_MIN, GAMMA)

    # Check average cost share equals TARGET when v=1 and theta~uniform
    grid = np.linspace(0, 1, 10001)
    shares = c(grid, 1.0)  # since v=1, this is the share itself
    avg_share = np.trapz(shares, grid) / (grid[-1] - grid[0])
    assert np.isclose(avg_share, TARGET, atol=5e-4)

    # Monotone decreasing in theta for fixed v
    v = 2.0
    s = c(grid, v)
    assert np.all(np.diff(s) <= 1e-10)
