import numpy as np
from .lomb_scargle import lomb_scargle


def lomb_scargle_multiband(t, y, dy, filt, omegas, Nterms=1):
    """
    (note: always fit offset)
    """
    t, y, dy, filt = map(np.asarray, (t, y, dy, filt))
    try:
        b = np.broadcast(t, y, dy, filt)
    except:
        raise ValueError("Inputs are incompatible shapes")

    # Perform a separate Lomb-Scargle for each band
    masks = np.array([(filt == f) for f in np.unique(filt)])
    powers = np.array([lomb_scargle(t[mask], y[mask], dy[mask],
                                    omegas, Nterms=Nterms,
                                    center_data=False, fit_offset=True)
                       for mask in masks])

    # Return sum of powers weighted by chi2-normalization
    chi2_0 = np.array([np.sum(y[mask] ** 2) for mask in masks])
    return np.dot(chi2_0 / chi2_0.sum(), powers)
    
