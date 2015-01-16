import numpy as np
from numpy.testing import assert_allclose

from ..modeler import PeriodicModeler


class FakeModeler(PeriodicModeler):
    """Fake periodic modeler for testing PeriodicModeler base"""
    def _fit(self, t, y, dy, filts):
        pass

    def _predict(self, t, filts, period):
        return np.ones(len(t))

    def _score(self, periods):
        return np.exp(-np.abs(periods - np.round(periods)))


def test_modeler_base():
    """Smoke-test of PeriodicModeler base class"""
    t = np.linspace(0, 10, 100)
    y = np.random.rand(len(t))
    dy = 0.1
    filts = None

    model = FakeModeler()

    # test setting the period range for the optimizer
    model.optimizer.period_range = (0.8, 1.2)

    # test fitting the model
    model.fit(t, y, dy, filts)

    # test the score function
    assert_allclose(model.score(1.0), 1.0)

    # test the best_period property
    assert_allclose(model.best_period, 1, rtol=1E-4)

    # test the predict function
    assert_allclose(model.predict(t, filts), 1)

    
