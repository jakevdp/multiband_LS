from __future__ import division, print_function
__all__ = ['LombScargle', 'LombScargleAstroML',
           'LombScargleMultiband', 'LombScargleMultibandFast']

import warnings

import numpy as np
from scipy import optimize


# TODO: there should be an option on fit() to automatically scan frequencies
#       and find a suitable omega... then the predict() and best_params()
#       methods could use this.


class PeriodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PeriodicModeler")

    def fit(self, t, y, dy, filts=None):
        raise NotImplementedError()

    def periodogram(self, omegas):
        raise NotImplementedError()

    def best_params(self, omega):
        raise NotImplementedError()

    def predict(self, t, omega):
        raise NotImplementedError()

    def __call__(self, omegas):
        return self.periodogram(omegas)

    def find_best_omega(self, omega_min=1, omega_max=60, Nzooms=10, verbose=1):
        """Find the best value of omega for the given data.

        This method attempts to be smart: it uses the range of the fit to
        estimate the expected peak widths, and chooses a resolution so that
        peaks will not be missed. Finally, it zooms-in on the top N peaks
        to find an accurate estimate of the location of highest power.

        Parameters
        ----------
        omega_min : float
            Minimum angular frequency of the scan range
        omega_max : float
            Maximum angular frequency of the scan range
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

        # Make sure things are in the right order
        omega_min, omega_max = np.sort([omega_min, omega_max])

        t = np.asarray(self.fit_data_['t'])
        expected_width = (2 * np.pi / (t.max() - t.min()))
        omega_step = 0.2 * expected_width
        omegas = np.arange(omega_min, omega_max, omega_step)

        if verbose:
            print("- Computing periods at {0:.0f} steps".format(len(omegas)))

        P = self.periodogram(omegas)

        # Choose the top ten peaks and zoom-in on them
        i = np.argsort(P)[-Nzooms:]
        omegas = np.concatenate([np.linspace(omega - 3 * omega_step,
                                             omega + 3 * omega_step, 500)
                                 for omega in omegas[i]])
        if verbose:
            print("- Zooming & computing periods at {0:.0f} furthersteps"
                  "".format(len(omegas)))

        P = self.periodogram(omegas)
        return omegas[np.argmax(P)]

    def find_best_period(self, P_min=0.2, P_max=1.2, Nzooms=10, verbose=1):
        """Find the best period for the given data.

        This method attempts to be smart: it uses the range of the fit to
        estimate the expected peak widths, and chooses a resolution so that
        peaks will not be missed. Finally, it zooms-in on the top N peaks
        to find an accurate estimate of the location of highest power.

        Parameters
        ----------
        P_min : float
            Minimum period of the scan range
        P_max : float
            Maximum period of the scan range
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
        omega_best = self.find_best_omega(omega_min=2 * np.pi / P_max,
                                          omega_max=2 * np.pi / P_min,
                                          Nzooms=Nzooms, verbose=verbose)
        return 2 * np.pi / omega_best


class LombScargle(PeriodicModeler):
    """Lomb-Scargle Periodogram Implementation

    This is a generalized periodogram implementation using the matrix formalism
    outlined in VanderPlas & Ivezic 2015. It allows computation of periodograms
    and best-fit models for both the classic normalized periodogram and
    truncated Fourier series generalizations.

    Parameters
    ----------
    center_data : boolean (default = True)
        If True, then compute the weighted mean of the input data and subtract
        before fitting the model.
    fit_offset : boolean (default = True)
        If True, then fit a floating-mean sinusoid model.
    Nterms : int (default = 1)
        Number of Fourier frequencies to fit in the model
    regularization : float, vector or None (default = None)
        If specified, then add this regularization penalty to the
        least squares fit.
    regularize_by_trace : boolean (default = True)
        If True, multiply regularization by the trace of the matrix

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargle().fit(t, y, dy)
    >>> omega_best = ls.find_best_omega()
    >>> omega_best
    10.000707659055472
    >>> ls.predict(t=0, omega=omega_best)
    array(-0.012038993444193624)

    See Also
    --------
    LombScargleAstroML
    LombScargleMultiband
    LombScargleMultibandFast
    """
    def __init__(self, center_data=True, fit_offset=True, Nterms=1,
                 regularization=None, regularize_by_trace=True):
        self.center_data = center_data
        self.fit_offset = fit_offset
        self.Nterms = int(Nterms)
        self.regularization = regularization
        self.regularize_by_trace = regularize_by_trace

        if not self.center_data and not self.fit_offset:
            warnings.warn("Not centering data or fitting offset can lead "
                          "to poor results")

        if self.Nterms < 0:
            raise ValueError("Nterms must be non-negative")

        if self.Nterms == 0 and not fit_offset:
            raise ValueError("You're specifying an empty model, dude!")

    def _construct_X(self, omega, weighted=True, **kwargs):
        """Construct the design matrix for the problem"""
        t = kwargs.get('t', self.fit_data_['t'])
        dy = kwargs.get('dy', self.fit_data_['dy'])
        fit_offset = kwargs.get('fit_offset', self.fit_offset)

        if fit_offset:
            offsets = [np.ones(len(t))]
        else:
            offsets = []

        cols = sum(([np.sin((i + 1) * omega * t),
                     np.cos((i + 1) * omega * t)]
                    for i in range(self.Nterms)), offsets)

        if weighted:
            return np.transpose(np.vstack(cols) / dy)
        else:
            return np.transpose(np.vstack(cols))

    def _construct_X_M(self, omega, **kwargs):
        """Construct the weighted normal matrix of the problem"""
        X = self._construct_X(omega, weighted=True, **kwargs)
        M = np.dot(X.T, X)

        if self.regularization is not None:
            diag = M.ravel(order='K')[::M.shape[0] + 1]
            if self.regularize_by_trace:
                diag += diag.sum() * np.asarray(self.regularization)
            else:
                diag += np.asarray(self.regularization)

        return X, M

    def _compute_ymean(self, **kwargs):
        """Compute the (weighted) mean of the y data"""
        y = kwargs.get('y', self.fit_data_['y'])
        dy = kwargs.get('dy', self.fit_data_['dy'])

        y = np.asarray(y)
        dy = np.asarray(dy)

        if dy.size == 1:
            # if dy is a scalar, we use the simple mean
            return np.mean(y)
        else:
            w = 1 / dy ** 2
            return np.dot(y, w) / w.sum()

    def _construct_y(self, weighted=True, **kwargs):
        y = kwargs.get('y', self.fit_data_['y'])
        dy = kwargs.get('dy', self.fit_data_['dy'])
        center_data = kwargs.get('center_data', self.center_data)

        y = np.asarray(y)
        dy = np.asarray(dy)

        if center_data:
            y = y - self._compute_ymean(y=y, dy=dy)

        if weighted:
            return y / dy
        else:
            return y

    def fit(self, t, y, dy=1.0, filts=None):
        """Fit the Periodogram model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like
            errors on observed values
        """
        self.fit_data_ = dict(t=t, y=y, dy=dy)
        self.yw_ = self._construct_y(weighted=True)
        self.ymean_ = self._compute_ymean()
        return self

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
        if not hasattr(self, 'fit_data_'):
            raise ValueError("Must call obj.fit() before obj.periodogram()")

        # To handle all possible shapes of input, we convert to array,
        # get the output shape, and flatten.
        omegas = np.asarray(omegas)
        output_shape = omegas.shape

        # Set up the reference chi2. Note that this entire function would
        # be much simpler if we did not allow center_data=False.
        # We keep it just to make sure our math is correct
        chi2_0 = np.dot(self.yw_.T, self.yw_)
        if self.center_data:
            chi2_ref = chi2_0
        else:
            yref = self._construct_y(weighted=True, center_data=True)
            chi2_ref = np.dot(yref.T, yref)
        chi2_0_minus_chi2 = np.zeros(omegas.size, dtype=float)

        # Iterate through the omegas and compute the power for each
        for i, omega in enumerate(omegas.flat):
            Xw, XTX = self._construct_X_M(omega)
            XTy = np.dot(Xw.T, self.yw_)
            chi2_0_minus_chi2[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

        # construct and return the power from the chi2 difference
        if self.center_data:
            P = chi2_0_minus_chi2 / chi2_ref
        else:
            P = 1 + (chi2_0_minus_chi2 - chi2_0) / chi2_ref

        return P.reshape(output_shape)

    def best_params(self, omega):
        """Compute the maximum likelihood model parameters at frequency omega

        Parameters
        ----------
        omega : float
            The angular frequency at which to compute the best parameters

        Returns
        -------
        theta : np.ndarray
            The array of model parameters for the best-fit model at omega
        """
        Xw, XTX = self._construct_X_M(omega)
        XTy = np.dot(Xw.T, self.yw_)
        return np.linalg.solve(XTX, XTy)

    def predict(self, t, omega):
        """Compute the best-fit model at ``t`` for a given frequency omega

        Parameters
        ----------
        omega : float
            The angular frequency at which to compute the best parameters
        t : float or array_like
            times at which to predict

        Returns
        -------
        y : np.ndarray
            predicted model values at times t
        """
        t = np.asarray(t)
        outshape = t.shape
        theta = self.best_params(omega)
        X = self._construct_X(omega, weighted=False, t=t.ravel())
        return np.reshape(self.ymean_ + np.dot(X, theta), outshape)


class LombScargleAstroML(PeriodicModeler):
    """Lomb-Scargle Periodogram Implementation using AstroML

    This is a generalized periodogram implementation which uses the periodogram
    functions from astroML. Compared to LombScargle, this implementation is
    both faster and more memory-efficient.

    Parameters
    ----------
    center_data : boolean (default = True)
        If True, then compute the weighted mean of the input data and subtract
        before fitting the model.
    fit_offset : boolean (default = True)
        If True, then fit a floating-mean sinusoid model.
    slow_version : boolean (default = False)
        If True, use the slower pure-python version from astroML. Otherwise,
        use the faster version of the code from astroML_addons

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargleAstroML().fit(t, y, dy)
    >>> omega_best = ls.find_best_omega()
    >>> omega_best
    10.000707659055472

    See Also
    --------
    LombScargle
    LombScargleMultiband
    LombScargleMultibandFast
    """
    def __init__(self, fit_offset=True, center_data=True,
                 slow_version=False):
        if slow_version:
            from astroML.time_series._periodogram import lomb_scargle
        else:
            from astroML.time_series import lomb_scargle
        self._LS_func = lomb_scargle
        self.fit_offset = fit_offset
        self.center_data = center_data

    def fit(self, t, y, dy, filts=None):
        """Fit the Periodogram model to the data.

        Parameters
        ----------
        t : array_like, one-dimensional
            sequence of observation times
        y : array_like, one-dimensional
            sequence of observed values
        dy : float or array_like
            errors on observed values
        """
        self.fit_data_ = dict(t=t, y=y, dy=dy)
        return self

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
        t = self.fit_data_['t']
        y = self.fit_data_['y']
        dy = self.fit_data_['dy']
        return self._LS_func(t, y, dy, omegas,
                             generalized=self.fit_offset,
                             subtract_mean=self.center_data)


class LombScargleMultibandFast(PeriodicModeler):
    def __init__(self, Nterms=1, BaseModel=LombScargle):
        # Note: center_data must be True, or else the chi^2 weighting will fail
        self.Nterms = Nterms
        self.BaseModel = BaseModel

    def fit(self, t, y, dy, filts):
        t, y, dy, filts = np.broadcast_arrays(t, y, dy, filts)
        self.unique_filts_ = np.unique(filts)
        masks = [(filts == f) for f in self.unique_filts_]
        self.models_ = [self.BaseModel(Nterms=self.Nterms, center_data=True,
                                       fit_offset=True).fit(t[m], y[m], dy[m])
                        for m in masks]
        return self

    def periodogram(self, omegas):
        # Return sum of powers weighted by chi2-normalization
        powers = np.array([model.periodogram(omegas)
                           for model in self.models_])
        chi2_0 = np.array([np.sum(model.yw_ ** 2) for model in self.models_])
        return np.dot(chi2_0 / chi2_0.sum(), powers)

    def best_params(self, omega):
        return np.asarray([model.best_params(omega) for model in self.models_])

    def predict(self, t, filts, omega):
        vals = set(np.unique(filts))
        if not vals.issubset(self.unique_filts_):
            raise ValueError("filts does not match training data: "
                             "{0}".format(self.unique_filts - vals))

        t, filts = np.broadcast_arrays(t, filts)

        result = np.zeros(t.shape, dtype=float)
        masks = ((filts == f) for f in self.unique_filts_)
        for model, mask in zip(self.models_, masks):
            result[mask] = model.predict(t[mask], omega)
        return result


class LombScargleMultiband(LombScargle):
    def __init__(self, Nterms_base=1, Nterms_band=1,
                 reg_base=None, reg_band=1E-6, regularize_by_trace=True,
                 center_data=True):
        self.Nterms_base = Nterms_base
        self.Nterms_band = Nterms_band
        self.reg_base = reg_base
        self.reg_band = reg_band
        self.regularize_by_trace = regularize_by_trace
        self.center_data = center_data

    def fit(self, t, y, dy, filts):
        t, y, dy, filts = np.broadcast_arrays(t, y, dy, filts)
        self.fit_data_ = dict(t=t, y=y, dy=dy, filts=filts)
        self.unique_filts_ = np.unique(filts)
        self.ymean_ = self._compute_ymean()

        masks = [(filts == filt) for filt in self.unique_filts_]
        self.ymean_by_filt_ = [LombScargle._compute_ymean(self,
                                                          y=y[mask],
                                                          dy=dy[mask])
                               for mask in masks]

        self.yw_ = self._construct_y(weighted=True)
        self.regularization = self._construct_regularization()
        return self

    def _construct_regularization(self):
        if self.reg_base is None and self.reg_band is None:
            reg = 0
        else:
            Nbase = 1 + 2 * self.Nterms_base
            Nband = 1 + 2 * self.Nterms_band
            reg = np.zeros(Nbase + len(self.unique_filts_) * Nband)
            if self.reg_base is not None:
                reg[:Nbase] = self.reg_base
            if self.reg_band is not None:
                reg[Nbase:] = self.reg_band
        return reg

    def _compute_ymean(self, **kwargs):
        y = kwargs.get('y', self.fit_data_['y'])
        dy = kwargs.get('dy', self.fit_data_['dy'])
        filts = kwargs.get('filts', self.fit_data_['filts'])

        ymean = np.zeros(y.shape)
        for filt in np.unique(filts):
            mask = (filts == filt)
            ymean[mask] = LombScargle._compute_ymean(self, y=y[mask],
                                                     dy=dy[mask])
        return ymean

    def _construct_y(self, weighted=True, **kwargs):
        y = kwargs.get('y', self.fit_data_['y'])
        dy = kwargs.get('dy', self.fit_data_['dy'])
        filts = kwargs.get('filts', self.fit_data_['filts'])
        center_data = kwargs.get('center_data', self.center_data)

        ymean = self._compute_ymean(**kwargs)

        if center_data:
            y = y - ymean

        if weighted:
            return y / dy
        else:
            return y

    def _construct_X(self, omega, weighted=True, **kwargs):
        t = kwargs.get('t', self.fit_data_['t'])
        dy = kwargs.get('dy', self.fit_data_['dy'])
        filts = kwargs.get('filts', self.fit_data_['filts'])

        # X is a huge-ass matrix that has lots of zeros depending on
        # which filters are present...
        #
        # huge-ass, quantitatively speaking, is of shape
        #  [len(t), (1 + 2 * Nterms_base + nfilts * (1 + 2 * Nterms_band))]

        # TODO: this could be more efficient
        cols = [np.ones(len(t))]
        cols = sum(([np.sin((i + 1) * omega * t),
                     np.cos((i + 1) * omega * t)]
                    for i in range(self.Nterms_base)), cols)

        for filt in self.unique_filts_:
            cols.append(np.ones(len(t)))
            cols = sum(([np.sin((i + 1) * omega * t),
                         np.cos((i + 1) * omega * t)]
                        for i in range(self.Nterms_band)), cols)
            mask = (filts == filt)
            for i in range(-1 - 2 * self.Nterms_band, 0):
                cols[i][~mask] = 0

        if weighted:
            return np.transpose(np.vstack(cols) / dy)
        else:
            return np.transpose(np.vstack(cols))

    def predict(self, t, filts, omega):
        vals = set(np.unique(filts))
        if not vals.issubset(self.unique_filts_):
            raise ValueError("filts does not match training data: "
                             "{0}".format(self.unique_filts - vals))

        t, filts = np.broadcast_arrays(t, filts)
        output_shape = t.shape

        t = t.ravel()
        filts = filts.ravel()

        # TODO: broadcast this
        ymeans = np.zeros(len(filts))
        for i, filt in enumerate(filts):
            j = np.where(self.unique_filts_ == filt)[0][0]
            ymeans[i] = self.ymean_by_filt_[j]

        theta = self.best_params(omega)
        X = self._construct_X(omega, weighted=False, t=t, filts=filts)
        return (ymeans + np.dot(X, theta)).reshape(output_shape)
