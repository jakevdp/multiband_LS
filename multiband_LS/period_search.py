import numpy as np
from astroML.time_series import lomb_scargle


def _single_curve(t, y, dy, i):
    t = t[:, i]
    y = y[:, i]
    dy = dy[:, i]

    good = ~(np.isnan(t) | np.isnan(y) | np.isnan(dy))

    return (t[good], y[good], dy[good])    


def period_search(t, y, dy, pmin=0.2, pmax=1.0, N=50000):
    """Search periods band-by-band"""
    # do a coarse-grained zoom for the first pass
    data = [_single_curve(t, y, dy, i) for i in range(5)]
    period = np.linspace(pmin, pmax, N)
    dp = period[1] - period[0]

    p_best = []
    for i, filt in enumerate('ugriz'):
        # shift period for greater coverage
        omega = 2 * np.pi / (period + 0.2 * i * dp)
        t, y, dy = data[i]
        P = lomb_scargle(t, y, dy, omega)
        p_best.append(period[np.argmax(P)])

    # now zoom-in around the best periods
    p_best = np.unique(0.001 * (np.array(p_best) // 0.001))
    #print(p_best)

    period = np.concatenate([np.linspace(p - 0.05, p + 0.05, 10000)
                             for p in p_best])
    omega = 2 * np.pi / period
    
    p_best = []
    for i, filt in enumerate('ugriz'):
        t, y, dy = data[i]
        P = lomb_scargle(t, y, dy, omega)
        p_best.append(period[np.argmax(P)])

    return np.array(p_best)
