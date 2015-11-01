import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()
seaborn.set_palette('deep', 5)

from gatspy.datasets import RRLyraeGenerated
from gatspy.periodic import LombScargle, LombScargleMultiband


# Choose a Sesar 2010 object to base our fits on
lcid = 1019544
rrlyrae = RRLyraeGenerated(lcid, random_state=0)
print("Extinction A_r = {0:.4f}".format(rrlyrae.obsmeta['rExt']))

# Generate data in a 6-month observing season
Nobs = 30
rng = np.random.RandomState(0)

nights = np.arange(6 * Nobs)
rng.shuffle(nights)
nights = nights[:Nobs]

t = 57000 + nights + 0.05 * rng.randn(Nobs)
dy = 0.06 + 0.01 * rng.randn(Nobs)
mags = np.array([rrlyrae.generated(band, t, err=dy, corrected=False)
                 for band in 'ugriz'])
filts = np.array([f for f in 'ugriz'])


def plot_data(ax, t, y, dy, f):
    for i, band in enumerate(filts):
        mask = (f == band)
        ax.errorbar((t[mask] / rrlyrae.period) % 1, y[mask], dy[mask],
                    fmt='.', label=band)
    ax.set_ylim(18, 14.5)
    #ax.legend(loc='upper left', fontsize=10, ncol=3)
    ax.set_xlabel('phase')
    ax.set_ylabel('magnitude')


period = rrlyrae.period
tfit = np.linspace(0, period, 1000)
periods = np.linspace(0.2, 1.2, 5000)


for i in range(2):
    fig, ax = plt.subplots(2, 2, figsize=(10, 4), sharex='col')
    fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1, bottom=0.15)

    if i == 0:
        ind = np.arange(Nobs) % 5
        y = mags[ind, np.arange(Nobs)]
        f = np.array(filts)[ind]
    else:
        arrs = np.broadcast_arrays(t, mags, dy, filts[:, None])
        t, y, dy, f = map(np.ravel, arrs)

    model1 = LombScargle()
    model1.fit(t, y, dy)
    yfit = model1.predict(tfit, period=period)

    plot_data(ax[0, 0], t, y, dy, f)
    ax[0, 0].plot(tfit / period, yfit, '-', color='gray', lw=4, alpha=0.5)
    ax[0, 1].plot(periods, model1.score(periods), color='gray')
    ax[0, 0].set_xlabel('')

    model2 = LombScargleMultiband(Nterms_base=1, Nterms_band=1)
    model2.fit(t, y, dy, f)
    yfits = model2.predict(tfit, filts=filts[:, None], period=period)

    plot_data(ax[1, 0], t, y, dy, f)
    for j in range(5):
        ax[1, 0].plot(tfit / period, yfits[j])
    ax[1, 1].plot(periods, model2.score(periods))

    fig.savefig('naive_{0}.png'.format(i + 1))

plt.show()
