"""
Supersmoother code for periodic modeling
"""
from __future__ import print_function, division

import numpy as np

try:
    import supersmoother as ssm
except ImportError:
    raise ImportError("Package supersmoother is required. "
                      "Use ``pip install supersmoother`` to install")

from .modeler import PeriodicModeler


class SuperSmoother(PeriodicModeler):
    """Periodogram based on Friedman's SuperSmoother.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ssm = SuperSmoother().fit(t, y, dy)
    >>> ssm.best_period
    0.62826749832108475
    >>> ssm.score(ls.best_period)
    array(0.9951882158877049)
    >>> ssm.predict([0, 0.5])
    array([ 0.06759746, -0.90006247])

    See Also
    --------
    LombScargle
    """
    def __init__(self, optimizer=None):
        PeriodicModeler.__init__(self, optimizer)

    def _fit(self, t, y, dy, filts):
        if filts is not None:
            raise NotImplementedError("``filts`` keyword is not supported")

        # TODO: this should actually be a weighted median, probably...
        mu = np.sum(y / dy ** 2) / np.sum(1 / dy ** 2)
        self.baseline_err = np.mean(abs((y - mu) / dy))

    def _predict(self, t, filts, period):
        model = ssm.SuperSmoother().fit(self.t % period, self.y, self.dy)
        return model.predict(t % period)

    def _score(self, periods):
        # double-up the data to account for periodicity.
        # TODO: push periodicity to the supersmoother package
        N = len(self.t)
        N4 = N // 4
        t = np.concatenate([self.t, self.t])
        y = np.concatenate([self.y, self.y])
        dy = np.concatenate([self.dy, self.dy])

        results = []
        for p in periods:
            # compute doubled phase and sort
            phase = t % p
            phase[N:] += p
            isort = np.argsort(phase)[N4: N + 3 * N4]
            phase = phase[isort]
            yp = y[isort]
            dyp = dy[isort]

            # compute model
            model = ssm.SuperSmoother().fit(phase, yp, dyp, presorted=True)

            # take middle residuals
            resids = model.cv_residuals()[N4: N4 + N]
            results.append(1 - np.mean(np.abs(resids)) / self.baseline_err)

        return np.asarray(results)
        
