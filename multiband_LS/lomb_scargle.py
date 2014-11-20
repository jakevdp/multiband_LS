from __future__ import division, print_function

import warnings

import numpy as np
from scipy import optimize


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
        Xw = self._construct_Xw(omega, weighted=True)
        XTX = np.dot(Xw.T, Xw)
        XTy = np.dot(Xw.T, self.yw_)
        return np.linalg.solve(XTX, XTy)

    def predict(self, t, omega):
        theta = self.best_params(omega)
        X = self._construct_Xw(omega, weighted=False, t=t)
        return self.ymean_ + np.dot(X, theta)
        


def _construct_X(t, dy, omega, Nterms=1, fit_offset=False):
    if fit_offset:
        offsets = [np.ones(len(t))]
    else:
        offsets = []

    cols = sum(([np.sin((i + 1) * omega * t),
                 np.cos((i + 1) * omega * t)]
                for i in range(Nterms)), offsets)

    return np.transpose(np.vstack(cols) / dy)


def _construct_y(y, dy, center_data=True):
    if center_data:
        w = 1. / dy ** 2
        mean = np.dot(y, w) / w.sum()
        y = y - mean
    return y / dy


def best_params(t, y, dy, omega, Nterms=1, fit_offset=False):
    t, y, dy, omega = map(np.asarray, (t, y, dy, omega))

    Xw = _construct_X(t, dy, omega, Nterms, fit_offset)
    yw = _construct_y(y, dy, fit_offset)

    XTX = np.dot(Xw.T, Xw)
    XTy = np.dot(Xw.T, yw)

    return np.linalg.solve(XTX, XTy)


def lomb_scargle(t, y, dy, omegas, Nterms=1,
                 center_data=True, fit_offset=True):
    # P = (chi2_0 - chi2) / chi2_ref
    # chi2_0 = y^T y
    # chi2_ref = (y - mean)^T (y - mean)
    # so if center_data is True, then chi2_ref == chi2_0

    t, y, dy, omegas = map(np.asarray, (t, y, dy, omegas))

    yw = _construct_y(y, dy, center_data)
    chi2_0 = np.dot(yw.T, yw)

    if center_data:
        chi2_ref = chi2_0
    else:
        yref = _construct_y(y, dy, True)
        chi2_ref = np.dot(yref.T, yref)

    P_LS = np.zeros_like(omegas)

    for i, omega in enumerate(omegas):
        Xw = _construct_X(t, dy, omega, Nterms, fit_offset)
        XTX = np.dot(Xw.T, Xw)
        XTy = np.dot(Xw.T, yw)
        P_LS[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

    P_LS -= chi2_0
    P_LS /= chi2_ref
    P_LS += 1
    return P_LS


def window_power(t, dy, omegas, **kwargs):
    kwargs['center_data'] = False
    kwargs['fit_offset'] = False
    return lomb_scargle(t, np.ones_like(t), dy, omegas, **kwargs)


def _construct_X_multiband(t, dy, omega, Nterms, fit_offset,
                           filter_ids, Nterms_multiband):
    if fit_offset:
        cols = [np.ones(len(t))]
    else:
        cols = []

    cols = sum(([np.sin((i + 1) * omega * t),
                 np.cos((i + 1) * omega * t)]
                for i in range(Nterms)), cols)

    uniquefilts = np.unique(filter_ids)

    for filt in uniquefilts[1:]:
        mask = (filter_ids == filt)

        # offset term
        cols += [mask.astype(float)]
        
        for j in range(Nfilts):
            sin_terms = np.zeros_like(t)
            cos_terms = np.zeros_like(t)
            sin_terms[mask] = np.sin((j + 1) * omega * t[mask])
            cos_terms[mask] = np.cos((j + 1) * omega * t[mask])
            cols += [sin_terms, cos_terms]

    return np.transpose(np.vstack(cols) / dy)


def _construct_y_multiband(y, dy, filter_ids, center_filt=0):
    if center is not None:
        mask = (filter_ids == center_filt)
        w = 1. / dy[mask] ** 2
        mean = np.dot(y[mask], w) / w.sum()
        y = y - mean
    return y / dy


def _compute_chi2_ref_multiband(y, dy, filter_ids, unique_filters):
    yw = np.array(y, copy=True)
    w = 1. / dy ** 2
    for filt in unique_filters:
        mask = (filter_ids == filt)
        mean = np.dot(y[mask], w[mask]) / w[mask].sum()
        yw[mask] -= mean
    yw /= dy
    return np.dot(yw, yw)


def best_params_multiband(t, y, dy, omega, Nterms=1, fit_offset=False,
                          filter_ids=None, Nterms_multiband=0):
    # If filter_ids are not provided, this is just a normal lomb-scargle
    if filter_ids is None:
        return best_params(t, y, dy, omegas, Nterms, fit_offset)

    t, y, dy, filter_ids = map(np.asarray, (t, y, dy, filter_ids))

    # normalize filter_ids
    unique_filters, filter_ids = np.unique(filter_ids, return_inverse=True)

    Xw = _construct_X_multiband(t, dy, omega, Nterms, fit_offset,
                                filter_ids, Nterms_multiband)
    yw = _construct_y(y, dy, filter_ids)

    XTX = np.dot(Xw.T, Xw)
    XTy = np.dot(Xw.T, yw)

    return np.linalg.solve(XTX, XTy)


def lomb_scargle_multiband(t, y, dy, omegas, Nterms=1, fit_offset=False,
                           filter_ids=None, Nterms_multiband=0):
    """Multiband Lomb-Scargle Periodogram"""
    # If filter_ids are not provided, this is just a normal lomb-scargle
    if filter_ids is None:
        return lomb_scargle(t, y, dy, omegas, Nterms, fit_offset)

    t, y, dy, filter_ids, omegas = map(np.asarray,
                                       (t, y, dy, filter_ids, omegas))

    # normalize filter_ids
    unique_filters, filter_ids = np.unique(filter_ids, return_inverse=True)

    yw = _construct_y_multiband(y, dy, filter_ids)
    chi2_0 = _compute_chi2_ref_multiband(y, dy, filter_ids, unique_filters)
    P_LS = np.zeros_like(omegas)

    for omega in omegas:
        Xw = _construct_X_multiband(t, y, dy, omega, Nterms, fit_offset,
                                    filter_ids, Nterms_multiband)
        XTX = np.dot(Xw.T, Xw)
        XTy = np.dot(Xw.T, yw)
        P_LS[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

    P_LS /= chi2_0
    return P_LS
