import numpy as np
import pytest

from optimal_ipr.distributions import value_distribution
from optimal_ipr.distributions.value_distribution import DATA_DIR, ZIP_FILENAME


def _data_available() -> bool:
    return (DATA_DIR / ZIP_FILENAME).exists()


@pytest.mark.skipif(not _data_available(), reason="KPSS patent value archive unavailable")
def test_value_distribution_properties():
    v_grid, v_weights = value_distribution()

    assert np.isclose(v_weights.sum(), 1.0)
    assert len(v_grid) == len(v_weights)
    assert np.all(np.diff(v_grid) >= 0)
    assert np.all(v_weights > 0)
    assert v_grid[0] > 0.1