import numpy as np
from numpy.testing import assert_allclose, assert_
from nose import SkipTest

from ..lomb_scargle import (LombScargle, LombScargleAstroML,
                            LombScargleMultiband,
                            LombScargleMultiband2)


def _generate_data(N=100, omega=10, theta=[10, 2, 3], dy=1, rseed=0):
    rng = np.random.RandomState(rseed)
    t = 20 * (2 * np.pi / omega) * rng.rand(N)
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)
    return t, y, dy


def test_lomb_scargle(N=100, omega=10):
    """Test whether the standard and generalized lomb-scargle
    give close to the same results for non-centered data"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)

    P1 = LombScargle(fit_offset=True).fit(t, y, dy).power(omegas)
    P2 = LombScargle(fit_offset=False).fit(t, y, dy).power(omegas)

    rms = np.sqrt(np.mean((P1 - P2) ** 2))
    assert_(rms < 0.005)


def test_lomb_scargle_multiband(N=100, omega=10):
    """Test that results are the same with/without filter labels"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)
    model = LombScargle(center_data=False, fit_offset=True)
    P_singleband = model.fit(t, y, dy).power(omegas)

    filts = np.ones(N)
    model = LombScargleMultiband()
    P_multiband = model.fit(t, y, dy, filts).power(omegas)
    assert_allclose(P_multiband, P_singleband)


def test_vs_astroML(N=100, omega=10):
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(omega - 4, omega + 4, 100)

    def compare_models(model1, model2):
        P = [model.fit(t, y, dy).power(omegas) for model in (model1, model2)]
        assert_allclose(P[0], P[1])

    # standard lomb-scargle
    for fit_offset in [True, False]:
        yield (compare_models,
               LombScargle(fit_offset=fit_offset),
               LombScargleAstroML(fit_offset=fit_offset))

    # Sanity check: make sure they work without centering data
    yield (compare_models,
           LombScargleAstroML(center_data=False),
           LombScargle(center_data=False))


def test_construct_X(N=100, omega=10):
    """
    Check whether the X array is constructed correctly
    """
    t, y, dy = _generate_data(N, omega)
    
    X = [LombScargle(N, fit_offset=False).fit(t, y, dy)._construct_X(omega)
         for N in [1, 2, 3]]
    Y = [LombScargle(N, fit_offset=True).fit(t, y, dy)._construct_X(omega)
         for N in [0, 1, 2, 3]]

    for i in range(3):
        assert_allclose(X[i], Y[i + 1][:, 1:])

    for i in range(4):
        assert_allclose(Y[i][:, 0], 1 / dy)

    for i in range(2):
        assert_allclose(X[i], X[i + 1][:, :2 * (i + 1)])


def test_best_params(N=100, omega=10):
    """Quick test for whether best params are computed without failure"""
    theta_true = [10, 2, 3]
    dy = 1.0

    t, y, dy = _generate_data(N, omega, theta_true, dy)

    for Nterms in [1, 2, 3]:
        model = LombScargle(Nterms=Nterms, center_data=False)
        model.fit(t, y, dy)
        theta_best = model.best_params(omega)
        assert_allclose(theta_true, theta_best[:3], atol=0.2)


def test_lomb_scargle_multiband2(N=100, omega=10):
    """Test that results are the same with/without filter labels"""
    t, y, dy = _generate_data(N, omega)
    omegas = np.linspace(1, omega + 1, 100)
    model = LombScargle(center_data=False, fit_offset=True)
    P_singleband = model.fit(t, y, dy).power(omegas)

    filts = np.ones(N)
    model_mb = LombScargleMultiband2(center_data=False)
    P_multiband = model_mb.fit(t, y, dy, filts).power(omegas)
    assert_allclose(P_multiband, P_singleband)
    assert_allclose(model.best_params(omega),
                    model_mb.best_params(omega)[:3])
