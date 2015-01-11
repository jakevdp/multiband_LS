from __future__ import division, print_function

try:
    import supersmoother as ssm
except ImportError:
    raise ImportError("Package supersmoother required "
                      "(``pip install supersmoother``)")

import numpy as np

from .base import PeriodicModeler


class SuperSmoother(PeriodicModeler):
    """SuperSmoother periodogram implementation

    See Also
    --------
    LombScargle
    """
    def __init__(self, period_range=(0.2, 1.1)):
        self.period_range = period_range

    def fit(self, t, y, dy=1.0, filts=None):
        """Fit the Supersmoother model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like
            errors on observed values
        """
        t, y, dy = np.broadcast_arrays(t, y, dy)
        self.fit_data_ = dict(t=t, y=y, dy=dy)

        # TODO: mu should actually be a weighted median, probably...
        mu = np.sum(y / dy ** 2) / np.sum(1 / dy ** 2)
        self.baseline_err = np.mean(abs((y - mu) / dy))

        return self

    def _predict(self, t, omega, filts=None):
        t_out = t
        period = 2 * np.pi / omega

        t, y, dy = (self.fit_data_[key] for key in ['t', 'y', 'dy'])

        N = len(t)
        N4 = N // 4
        N2 = N // 2
        t = np.concatenate([t, t])
        y = np.concatenate([y, y])
        dy = np.concatenate([dy, dy])
        phase = t % period
        phase[N: N + N2] += period
        phase[N + N2:] -= period
        isort = np.argsort(phase)[N4: -N4]
        phase, yp, dyp = phase[isort], y[isort], dy[isort]

        model = ssm.SuperSmoother().fit(phase, yp, dyp)
        return model.predict(t_out % period)

    def periodogram(self, omegas):
        """Compute the periodogram at the given angular frequencies

        Parameters
        ----------
        omegas : array_like
            Array of angular frequencies at which to compute
            the periodogram

        Returns
        -------
        periodogram : np.ndarray
            Array of normalized powers (between 0 and 1) for each frequency
        """
        period = 2 * np.pi / np.asarray(omegas)
        t, y, dy = (self.fit_data_[key] for key in ['t', 'y', 'dy'])

        # double-up the data to allow periodicity on the fits
        N = len(t)
        N4 = N // 4
        t = np.concatenate([t, t])
        y = np.concatenate([y, y])
        dy = np.concatenate([dy, dy])

        results = []
        for p in period.ravel():
            # compute doubled phase and sort
            phase = t % p
            phase[N:] += p
            isort = np.argsort(phase)[N4: -N4]
            phase, yp, dyp = phase[isort], y[isort], dy[isort]

            # compute model
            model = ssm.SuperSmoother().fit(phase, yp, dyp, presorted=True)

            # take middle residuals
            resids = model.cv_residuals()[N4: N4 + N]
            results.append(1 - np.mean(np.abs(resids)) / self.baseline_err)

        return np.asarray(results).reshape(period.shape)

        


        
