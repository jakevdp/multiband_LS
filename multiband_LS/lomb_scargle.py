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

    def fit(self, t, y, dy):
        raise NotImplementedError()

    def power(self, omegas):
        raise NotImplementedError()

    def best_params(self, omega):
        raise NotImplementedError()

    def predict(self, t, omega):
        raise NotImplementedError()


class LombScargle(PeriodicModeler):
    def __init__(self, center_data=True, fit_offset=True, Nterms=1):
        self.center_data = center_data
        self.fit_offset = fit_offset
        self.Nterms = int(Nterms)

        if not self.center_data and not self.fit_offset:
            warnings.warn("Not centering data or fitting offset can lead "
                          "to poor results")

        if self.Nterms < 0:
            raise ValueError("Nterms must be non-negative")

        if self.Nterms == 0 and fit_offset == False:
            raise ValueError("You're specifying an empty model, dude!")

    def _construct_X(self, omega, weighted=True, **kwargs):
        t = kwargs.get('t', self.fit_data_[0])
        dy = kwargs.get('dy', self.fit_data_[2])
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

    def _compute_ymean(self, **kwargs):
        y = kwargs.get('y', self.fit_data_[1])
        dy = kwargs.get('dy', self.fit_data_[2])
        w = 1. / dy ** 2
        return np.dot(y, w) / w.sum()

    def _construct_y(self, weighted=True, **kwargs):
        y = kwargs.get('y', self.fit_data_[1])
        dy = kwargs.get('dy', self.fit_data_[2])
        center_data = kwargs.get('center_data', self.center_data)

        if center_data:
            y = y - self.ymean_

        if weighted:
            return y / dy
        else:
            return y

    def fit(self, t, y, dy, filts=None):
        self.fit_data_ = (t, y, dy)
        self.ymean_ = self._compute_ymean()
        self.yw_ = self._construct_y(weighted=True)
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
            Xw = self._construct_X(omega, weighted=True)
            XTX = np.dot(Xw.T, Xw)
            XTy = np.dot(Xw.T, self.yw_)
            chi2_0_minus_chi2[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

        # construct and return the power from the chi2 difference
        if self.center_data:
            P = chi2_0_minus_chi2 / chi2_ref
        else:
            P =  1 + (chi2_0_minus_chi2 - chi2_0) / chi2_ref

        return P.reshape(output_shape)

    def best_params(self, omega):
        Xw = self._construct_X(omega, weighted=True)
        XTX = np.dot(Xw.T, Xw)
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
        self.fit_data_ = (t, y, dy)
        return self

    def power(self, omegas):
        t, y, dy = self.fit_data_
        return self._LS_func(t, y, dy, omegas,
                             generalized=self.fit_offset,
                             subtract_mean=self.center_data)


class LombScargleMultiband(PeriodicModeler):
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
            
