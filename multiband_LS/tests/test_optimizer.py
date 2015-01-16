"""Tests of the various optimizer classes"""
import numpy as np
from numpy.testing import assert_allclose

from ..optimizer import LinearScanOptimizer
from .. import LombScargle


def test_linear_scan():
    optimizer = LinearScanOptimizer(period_range=(0.8, 1.2))

    t = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * t)
    dy = 1

    model = LombScargle(optimizer=optimizer).fit(t, y, dy)

    # test candidate periods
    candidate_periods = optimizer._compute_candidate_periods(model)
    assert_allclose(candidate_periods.min(), 0.8, atol=0.02)
    assert_allclose(candidate_periods.max(), 1.2)

    # test finding best period
    best_period = optimizer.find_best_period(model)
    assert_allclose(best_period, 1, atol=1E-4)
