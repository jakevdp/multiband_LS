from __future__ import division, print_function

import sys

import numpy as np


class PeriodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PeriodicModeler")

    def fit(self, t, y, dy, filts=None):
        raise NotImplementedError()

    def score(self, period):
        raise NotImplementedError()

    def best_params(self, omega=None):
        """Compute the maximum likelihood model parameters at frequency omega

        Parameters
        ----------
        omega : float (optional)
            The angular frequency at which to compute the best parameters.
            if not specified, it will be computed via self.find_best_omega().

        Returns
        -------
        theta : np.ndarray
            The array of model parameters for the best-fit model at omega
        """
        if not hasattr(self, 'fit_data_'):
            raise ValueError("Must call fit() before periodogram/best_params")
        if omega is None:
            omega = self.best_omega
        return self._best_params(omega)

    def _best_params(self, omega):
        raise NotImplementedError()

    def predict(self, t, filts=None, omega=None):
        """Compute the best-fit model at ``t`` for a given frequency omega

        Parameters
        ----------
        t : float or array_like
            times at which to predict
        filts : array_like (optional)
            the array specifying the filter/bandpass for each observation. This
            is used only in multiband periodograms.
        omega : float (optional)
            The angular frequency at which to compute the best parameters.
            if not specified, it will be computed via self.find_best_omega().

        Returns
        -------
        y : np.ndarray
            predicted model values at times t
        """
        if not hasattr(self, 'fit_data_'):
            raise ValueError("Must call fit() before predict")
        if omega is None:
            omega = self.best_omega
        return self._predict(t=t, omega=omega, filts=filts)

    def _predict(self, t, omega, filts=None):
        raise NotImplementedError()

    def __call__(self, omegas):
        return self.periodogram(omegas)

    @property
    def best_omega(self):
        if not hasattr(self, '_best_omega'):
            self._best_omega = self.find_best_omega()
        return self._best_omega

    @property
    def best_period(self):
        return 2 * np.pi / self.best_omega

    def find_best_omega(self, omega_min=None, omega_max=None,
                        Nzooms=10, verbose=1):
        """Find the best value of omega for the given data.

        This method attempts to be smart: it uses the range of the fit to
        estimate the expected peak widths, and chooses a resolution so that
        peaks will not be missed. Finally, it zooms-in on the top N peaks
        to find an accurate estimate of the location of highest power.

        Parameters
        ----------
        omega_min : float (optional)
            Minimum angular frequency of the scan range. If not specified,
            then self.period_range will be used.
        omega_max : float (optional)
            Maximum angular frequency of the scan range. If not specified,
            then self.period_range will be used.
        Nzooms : integer
            Number of initial peaks to zoom on to find the final result
        verbose : boolean
            If True, then print diagnostics of the fits

        Returns
        -------
        omega_best : float
            The value of the angular frequency within the given range which
            corresponds to the maximum power.

        See Also
        --------
        find_best_period
        """
        if not hasattr(self, 'fit_data_'):
            raise ValueError("Must call fit() before find_best_period")
            
        if omega_min is None:
            omega_min = 2 * np.pi / max(self.period_range[0],
                                        self.period_range[1])
        if omega_max is None:
            omega_max = 2 * np.pi / min(self.period_range[0],
                                        self.period_range[1])

        # Make sure things are in the right order
        omega_min, omega_max = np.sort([omega_min, omega_max])

        t = np.asarray(self.fit_data_['t'])
        expected_width = (2 * np.pi / (t.max() - t.min()))
        omega_step = 0.2 * expected_width
        if verbose:
            print("Finding optimal frequency:")
            print(" - Using omega_step = {0:.5f}".format(omega_step))
            sys.stdout.flush()

        omegas = np.arange(omega_min, omega_max, omega_step)

        if verbose:
            print(" - Computing periods at {0:.0f} steps".format(len(omegas)))
            sys.stdout.flush()

        P = self.periodogram(omegas)

        # Choose the top ten peaks and zoom-in on them
        i = np.argsort(P)[-Nzooms:]
        omegas = np.concatenate([np.linspace(omega - 3 * omega_step,
                                             omega + 3 * omega_step, 500)
                                 for omega in omegas[i]])
        if verbose:
            print(" - Zooming & computing periods at {0:.0f} further steps"
                  "".format(len(omegas)))
            sys.stdout.flush()

        P = self.periodogram(omegas)
        return omegas[np.argmax(P)]

    def find_best_period(self, P_min=None, P_max=None, Nzooms=10, verbose=1):
        """Find the best period for the given data.

        This method attempts to be smart: it uses the range of the fit to
        estimate the expected peak widths, and chooses a resolution so that
        peaks will not be missed. Finally, it zooms-in on the top N peaks
        to find an accurate estimate of the location of highest power.

        Parameters
        ----------
        P_min : float (optional)
            Minimum period of the scan range. If not specified,
            then self.period_range will be used.
        P_max : float (optional)
            Maximum period of the scan range. If not specified,
            then self.period_range will be used.
        Nzooms : integer
            Number of initial peaks to zoom on to find the final result
        verbose : boolean
            If True, then print diagnostics of the fits

        Returns
        -------
        P_best : float
            The value of the period within the given range which
            corresponds to the maximum power.

        See Also
        --------
        find_best_omega
        """
        if P_min is None:
            P_min = min(self.period_range[0], self.period_range[1])
        if P_max is None:
            P_max = max(self.period_range[0], self.period_range[1])
        omega_best = self.find_best_omega(omega_min=2 * np.pi / P_max,
                                          omega_max=2 * np.pi / P_min,
                                          Nzooms=Nzooms, verbose=verbose)
        return 2 * np.pi / omega_best
