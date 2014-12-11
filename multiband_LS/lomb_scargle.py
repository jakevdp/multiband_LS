from __future__ import division, print_function
__all__ = ['LombScargle', 'LombScargleAstroML', 'LombScargleMultiband']

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

    def power(self, omegas):
        raise NotImplementedError()

    def best_params(self, omega):
        raise NotImplementedError()

    def predict(self, t, omega):
        raise NotImplementedError()

    def __call__(self, omegas):
        return self.power(omegas)

    def find_best_period(self, P_min=0.2, P_max=1.2, Nzooms=10, verbose=1):
        if not hasattr(self, 'fit_data_'):
            raise ValueError("Must call fit() before find_best_period")
        t = self.fit_data_['t']
        omega_min = 2 * np.pi / P_max
        omega_max = 2 * np.pi / P_min
        expected_width = (2 * np.pi / (t.max() - t.min()))
        omega_step = 0.2 * expected_width
        if verbose:
            print("- Computing periods at {0:.0f} "
                  "steps".format((omega_max - omega_min) // omega_step))
        omegas = np.arange(omega_min, omega_max, omega_step)
        P = self.power(omegas)

        # Choose the top ten peaks and zoom-in on them
        i = np.argsort(P)[-Nzooms:]
        omegas = np.concatenate([np.linspace(omega - 3 * omega_step,
                                             omega + 3 * omega_step,
                                             500) for omega in omegas[i]])
        P = self.power(omegas)
        return 2 * np.pi / omegas[np.argmax(P)]


class LombScargle(PeriodicModeler):
    def __init__(self, center_data=True, fit_offset=True, Nterms=1,
                 regularization=None, weight_by_diagonal=True):
        self.center_data = center_data
        self.fit_offset = fit_offset
        self.Nterms = int(Nterms)
        self.regularization = regularization
        self.weight_by_diagonal = weight_by_diagonal

        if not self.center_data and not self.fit_offset:
            warnings.warn("Not centering data or fitting offset can lead "
                          "to poor results")

        if self.Nterms < 0:
            raise ValueError("Nterms must be non-negative")

        if self.Nterms == 0 and fit_offset == False:
            raise ValueError("You're specifying an empty model, dude!")

    def _construct_X(self, omega, weighted=True, **kwargs):
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

    def _construct_X_M(self, omega, weighted=True, **kwargs):
        X = self._construct_X(omega, weighted, **kwargs)
        M = np.dot(X.T, X)

        if self.regularization is not None:
            diag = M.ravel(order='K')[::M.shape[0] + 1]
            if self.weight_by_diagonal:
                diag += diag.sum() * np.asarray(self.regularization)
            else:
                diag += np.asarray(self.regularization)

        return X, M

    def _compute_ymean(self, **kwargs):
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

    def fit(self, t, y, dy, filts=None):
        self.fit_data_ = dict(t=t, y=y, dy=dy)
        self.yw_ = self._construct_y(weighted=True)
        self.ymean_ = self._compute_ymean()
        return self

    def power(self, omegas):
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
            Xw, XTX = self._construct_X_M(omega, weighted=True)
            XTy = np.dot(Xw.T, self.yw_)
            chi2_0_minus_chi2[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

        # construct and return the power from the chi2 difference
        if self.center_data:
            P = chi2_0_minus_chi2 / chi2_ref
        else:
            P =  1 + (chi2_0_minus_chi2 - chi2_0) / chi2_ref

        return P.reshape(output_shape)

    def best_params(self, omega):
        Xw, XTX = self._construct_X_M(omega, weighted=True)
        XTy = np.dot(Xw.T, self.yw_)
        return np.linalg.solve(XTX, XTy)

    def predict(self, t, omega):
        theta = self.best_params(omega)
        X = self._construct_X(omega, weighted=False, t=t)
        return self.ymean_ + np.dot(X, theta)


class LombScargleAstroML(PeriodicModeler):
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
        self.fit_data_ = dict(t=t, y=y, dy=dy)
        return self

    def power(self, omegas):
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
        
    def power(self, omegas):
        # Return sum of powers weighted by chi2-normalization
        powers = np.array([model.power(omegas) for model in self.models_])
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
                 reg_base=None, reg_band=1E-6, weight_by_diagonal=True,
                 center_data=True):
        self.Nterms_base = Nterms_base
        self.Nterms_band = Nterms_band
        self.reg_base = reg_base
        self.reg_band = reg_band
        self.weight_by_diagonal = weight_by_diagonal
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
