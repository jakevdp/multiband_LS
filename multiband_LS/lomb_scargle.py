from __future__ import division, print_function

import numpy as np
from scipy import optimize


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
    t, y, dy, omegas = map(np.asarray, (t, y, dy, omegas))

    yw = _construct_y(y, dy, center_data)
    chi2_0 = np.dot(yw.T, yw)
    P_LS = np.zeros_like(omegas)

    for i, omega in enumerate(omegas):
        Xw = _construct_X(t, dy, omega, Nterms, fit_offset)
        XTX = np.dot(Xw.T, Xw)
        XTy = np.dot(Xw.T, yw)
        P_LS[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

    P_LS /= chi2_0
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
