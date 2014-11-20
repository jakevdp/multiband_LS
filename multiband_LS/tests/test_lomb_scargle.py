import numpy as np
from numpy.testing import assert_allclose, assert_
from nose import SkipTest

from ..lomb_scargle import lomb_scargle, best_params, _construct_X


def _generate_data(N=100, omega=10, dy=0.1, random_state=0):
    rng = np.random.RandomState(random_state)
    t = 10 * omega * rng.rand(N)
    y = 10 + 2 * np.sin(omega * t) + 3 * np.cos(omega * t - 0.3)
    dy = dy * (1 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_construct_X(N=100, omega=10):
    """
    Check whether the x array is constructed correctly
    """
    t, y, dy = _generate_data(N, omega)

    X1 = _construct_X(t, dy, omega, Nterms=1, fit_offset=False)
    assert_(X1.shape == (100, 2))

    X2 = _construct_X(t, dy, omega, Nterms=2, fit_offset=False)
    assert_(X2.shape == (100, 4))

    X3 = _construct_X(t, dy, omega, Nterms=2, fit_offset=True)
    assert_(X3.shape == (100, 5))

    assert_allclose(X1, X2[:, :2])
    assert_allclose(X2, X3[:, 1:])
    assert_allclose(X3[:, 0], 1. / dy)


def test_lomb_scargle(N=100, omega=10):
    """Test whether the standard and generalized lomb-scargle
    give close to the same results for non-centered data"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)

    P1 = lomb_scargle(t, y, dy, omegas, fit_offset=True)
    P2 = lomb_scargle(t, y, dy, omegas, fit_offset=False)

    rms = np.sqrt(np.mean((P1 - P2) ** 2))
    assert_(rms < 0.01)


def test_best_params(N=100, omega=10):
    """Quick test for whether best params are computed without failure"""
    # TODO: find a way to check these results!
    t, y, dy = _generate_data(N, omega)

    chi2_0 = y - np.dot(y, dy ** -2) / np.sum(dy ** -2)

    for fit_offset in [True, False]:
        for Nterms in [1, 2]:
            theta_best = best_params(t, y, dy, omega, Nterms, fit_offset)


def test_vs_astroML(N=100, omega=10):
    try:
        from astroML.time_series import lomb_scargle as lomb_scargle_astroML
    except ImportError:
        raise SkipTest("astroML is not available")

    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(omega - 4, omega + 4, 100)

    # standard lomb-scargle
    P1 = lomb_scargle_astroML(t, y, dy, omegas, generalized=False)
    P2 = lomb_scargle(t, y, dy, omegas, fit_offset=False)
    yield assert_allclose, P1, P2

    # generalized lomb-scargle
    P1 = lomb_scargle_astroML(t, y, dy, omegas, generalized=True)
    P2 = lomb_scargle(t, y, dy, omegas, fit_offset=True)
    yield assert_allclose, P1, P2

    # generalized lomb-scargle without first centering the data
    P2 = lomb_scargle(t, y, dy, omegas, fit_offset=True, center_data=False)
    yield assert_allclose, P1, P2
    

