import numpy as np

from optimal_ipr.fee import DEFAULT_FEE_M, build_fee_schedule


def test_shape_and_monotonicity():
    zeta = 0.04
    Z = build_fee_schedule(zeta)

    beta = np.linspace(0, 1, 101)
    v = 2.0
    fees = Z(beta, v)

    expected = (zeta * np.power(beta, DEFAULT_FEE_M)) * v
    assert np.allclose(fees, expected)

    # nonnegative, increasing in beta
    assert np.all(fees >= 0)
    assert np.all(np.diff(fees) >= -1e-12)

    # homogeneous of degree 1 in v
    assert np.isclose(Z(beta, 3.0), (3.0 / 2.0) * fees).all()
