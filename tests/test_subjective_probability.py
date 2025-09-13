import numpy as np
import pytest
from optimal_ipr.probability import build_subjective_probability


def test_subjective_probability_smoke():
    F = lambda x: x
    F_inv = lambda x: x
    p = build_subjective_probability(base_k=0.5, m_comp=25, F=F, F_inv=F_inv)

    th = np.linspace(0, 1, 11)
    # increasing in theta for fixed v (weakly, numerically)
    v = 2.0
    pv = p(th, v)
    assert pv.shape == th.shape
    assert np.all(np.isfinite(pv))
    assert 0.0 <= float(pv.min()) <= float(pv.max()) <= 1.0
