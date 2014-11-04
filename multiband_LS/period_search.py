import numpy as np
from astroML.time_series import lomb_scargle


def period_search(t, y, dy, pmin=0.2, pmax=1.2, N=10000):
    period = np.linspace(pmin, pmax, N)
    dp = period[1] - period[0]

    p_best = []

    for i, filt in enumerate('ugriz'):
        omega = 2 * np.pi / (period + 0.2 * i * dp)
        ti = t[:, i]
        yi = y[:, i]
        dyi = dy[:, i]

        good = ~(np.isnan(ti) | np.isnan(yi) | np.isnan(dyi))

        ti = ti[good]
        yi = yi[good]
        dyi = dyi[good]

        P = lomb_scargle(ti, yi, dyi, omega)

        p_best.append(period[np.argmax(P)])

    print(np.unique(p_best))

    # now zoom-in around the best periods
    zoomed_periods = np.concatenate([np.linspace(p - 0.05, p + 0.05, 10000)
                                     for p in np.unique(p_best)])
    omega = 2 * np.pi / zoomed_periods
    
    p_best = []
    for i, filt in enumerate('ugriz'):
        ti = t[:, i]
        yi = y[:, i]
        dyi = dy[:, i]

        good = ~(np.isnan(ti) | np.isnan(yi) | np.isnan(dyi))

        ti = ti[good]
        yi = yi[good]
        dyi = dyi[good]

        P = lomb_scargle(ti, yi, dyi, omega)
        p_best.append(zoomed_periods[np.argmax(P)])

    return p_best
