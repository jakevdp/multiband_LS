from __future__ import division, print_function

import numpy as np
from scipy import optimize


def _construct_X(t, dy, omega, Nterms=1, compute_offset=False):
    if compute_offset:
        offsets = [np.ones(len(t))]
    else:
        offsets = []

    cols = sum(([np.sin((i + 1) * omega * t),
                 np.cos((i + 1) * omega * t)]
                for i in range(Nterms)), offsets)

    return np.transpose(np.vstack(cols) / dy)


def _construct_y(y, dy, center=True):
    if center:
        w = 1. / dy ** 2
        y = y - (np.dot(y, w) / w.sum())
    return y / dy


def best_params(t, y, dy, omega, Nterms=1, compute_offset=False):
    t, y, dy, omega = map(np.asarray, (t, y, dy, omega))

    Xw = _construct_X(t, dy, omega, Nterms, compute_offset)
    yw = _construct_y(y, dy, compute_offset)

    XTX = np.dot(Xw.T, Xw)
    XTy = np.dot(Xw.T, yw)

    return np.linalg.solve(XTX, XTy)


def lomb_scargle(t, y, dy, omegas, Nterms=1, compute_offset=False):
    t, y, dy, omegas = map(np.asarray, (t, y, dy, omegas))

    yw = _construct_y(y, dy)
    chi2_0 = np.dot(yw.T, yw)
    P_LS = np.zeros_like(omegas)

    for i, omega in enumerate(omegas):
        Xw = _construct_X(t, dy, omega, Nterms, compute_offset)
        XTX = np.dot(Xw.T, Xw)
        XTy = np.dot(Xw.T, yw)
        P_LS[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

    P_LS /= chi2_0
    return P_LS
