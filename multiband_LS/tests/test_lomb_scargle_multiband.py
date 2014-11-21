import numpy as np
from numpy.testing import assert_allclose

from ..lomb_scargle import LombScargle
from ..lomb_scargle_multiband import lomb_scargle_multiband


def _generate_data(N=100, omega=10, dy=0.1, random_state=0):
    rng = np.random.RandomState(random_state)
    t = 10 * rng.rand(N)
    y = 10 + 2 * np.sin(omega * t) + 3 * np.cos(omega * t - 0.3)
    dy = dy * (1 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_lomb_scargle_multiband(N=100, omega=10):
    """Test that results are the same with/without filter labels"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)
    model1 = LombScargle(center_data=False, fit_offset=True)
    P_singleband = model1.fit(t, y, dy).power(omegas)

    filt = np.ones(N)
    P_multiband = lomb_scargle_multiband(t, y, dy, filt, omegas)
    assert_allclose(P_multiband, P_singleband)
    
