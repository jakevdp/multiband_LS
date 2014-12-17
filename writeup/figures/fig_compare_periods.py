import os, sys; sys.path.append(os.path.abspath('../..'))

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

from multiband_LS import LombScargleAstroML, LombScargleMultiband
from multiband_LS.memoize import CacheResults
from multiband_LS.data import fetch_light_curves
rrlyrae = fetch_light_curves()


def find_best_period_adhoc(lcid):
    t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
    models = [LombScargleAstroML(period_range=(0.2, 1.2)).fit(t[filts==filt],
                                                              y[filts==filt],
                                                              dy[filts==filt])
              for filt in 'ugriz']
    return np.array([model.best_period for model in models])


def compute_all_periods_adhoc():
    lcids = list(rrlyrae.ids)
    cache_dir = 'results_adhoc'
    cache = CacheResults(cache_dir, verbose=True)
    results = cache.call_iter(find_best_period_adhoc, lcids)
    return np.array(results)


def find_best_period(lcid, rrlyrae, Nterms_base=1, Nterms_band=0):
    t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
    LS = LombScargleMultiband(Nterms_base=Nterms_base,
                              Nterms_band=Nterms_band,
                              period_range=(0.2, 1.2))
    LS.fit(t, y, dy, filts)
    return LS.best_period


def compute_all_periods(Nterms_base, Nterms_band):
    lcids = list(rrlyrae.ids)
    cache_dir = 'results_{0}_{1}'.format(Nterms_base, Nterms_band)
    cache = CacheResults(cache_dir, verbose=True)
    return cache.call_iter(find_best_period, lcids,
                           args=(rrlyrae, Nterms_base, Nterms_band))


new_periods = compute_all_periods(1, 0)
old_periods = compute_all_periods(0, 1)
sesar_periods = np.array([rrlyrae.get_metadata(lcid)['P']
                                  for lcid in rrlyrae.ids])
adhoc_periods = mode(compute_all_periods_adhoc(), 1)[0]


fig, ax = plt.subplots()
ax.plot(sesar_periods, new_periods, 'o', alpha=0.5, ms=5)
#ax.plot(sesar_periods, adhoc_periods, 'o', alpha=0.5, ms=5)

P1 = np.linspace(0.1, 1.2)
ax.plot(P1, P1, ':k', alpha=0.7, lw=1, zorder=1)

for n in [1, 2, 3]:
    Pn = P1 / (1 + n * P1)
    ax.plot(P1, Pn, ':k', alpha=0.7, lw=1, zorder=1)
    ax.plot(Pn, P1, ':k', alpha=0.7, lw=1, zorder=1)

ax.set_xlabel('Sesar 2010 period (days)')
ax.set_ylabel('new-style period (days)')
ax.set_xlim(0.1, 1.2)
ax.set_ylim(0.1, 1.2)
plt.show()



if False:
    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax2)

    P1 = np.linspace(0.1, 1.2)
    P2 = P1 / (1 + P1)

    for ax in [ax1, ax2, ax3]:
        ax.plot(P1, P2, ':k', alpha=0.7, lw=1, zorder=1)
        ax.plot(P2, P1, ':k', alpha=0.7, lw=1, zorder=1)
        ax.set_xlim(0.1, 1.2)
        ax.set_ylim(0.1, 1.2)

    c = (new_periods - old_periods)
    kwds = dict(c=c, lw=0, cmap='RdBu', alpha=0.5, zorder=2)

    ax1.scatter(new_periods, sesar_periods, **kwds)
    ax1.set_ylabel('Sesar 2010')

    ax2.scatter(sesar_periods, old_periods, **kwds)
    ax2.set_xlabel('Sesar 2010')

    ax3.scatter(new_periods, old_periods, **kwds)
    ax3.set_xlabel('new')
    ax3.set_ylabel('old')

plt.show()
