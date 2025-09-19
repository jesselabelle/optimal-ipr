from __future__ import annotations
import numpy as np
from typing import Callable

DEFAULT_FEE_M: float = 1.1931736978417478


def build_fee_schedule(
    zeta: float, fee_M: float = DEFAULT_FEE_M
) -> Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]:
    """
    Return Z(beta, v) = (zeta * beta**fee_M) * v.

    Parameters
    ----------
    zeta : float
        Premium rate scalar.
    fee_M : float, default=DEFAULT_FEE_M
        Exponent on beta.

    Returns
    -------
    Z : callable
        Vectorized fee schedule in (beta, v).
    """
    def Z(beta, v):
        beta_arr = np.asarray(beta, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        return (zeta * np.power(beta_arr, fee_M)) * v_arr

    return Z
