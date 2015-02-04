"""
Here we plot a typical approach to the multi-band periodogram: treating each
band separately, and taking a majority vote between the bands.
"""
import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()

from gatspy.datasets import RRLyraeGenerated
from gatspy.periodic import (LombScargleAstroML, LombScargleMultiband,
                             NaiveMultiband)


# Choose a Sesar 2010 object to base our fits on
lcid = 1019544
rrlyrae = RRLyraeGenerated(lcid, random_state=0)
print("Extinction A_r = {0:.4f}".format(rrlyrae.obsmeta['rExt']))

# Generate data in a 6-month observing season
Nobs = 60
rng = np.random.RandomState(0)

nights = np.arange(180)
rng.shuffle(nights)
nights = nights[:Nobs]

# Find a subset of the simulated data. This is the same procedure as in
# fig_multiband_sim
t = 57000 + nights + 0.05 * rng.randn(Nobs)
dy = 0.06 + 0.01 * rng.randn(Nobs)
mags = np.array([rrlyrae.generated(band, t, err=dy, corrected=False)
                 for band in 'ugriz'])
filts = np.array([f for f in 'ugriz'])

# Here's our subset
filts = np.take(list('ugriz'), np.arange(Nobs), mode='wrap')
mags = mags[np.arange(Nobs) % 5, np.arange(Nobs)]
masks = [(filts == band) for band in 'ugriz']

fig, ax = plt.subplots(5, sharex=True, sharey=True)
fig.subplots_adjust(left=0.1, right=0.93, hspace=0.1)

periods = np.linspace(0.2, 1.4, 1000)

combos = [(1, 0), (0, 1), (2, 0), (2, 1), (2, 2)]

for axi, (Nbase, Nband) in zip(ax, combos):
    LS_multi = LombScargleMultiband(Nterms_base=Nbase,
                                    Nterms_band=Nband)
    LS_multi.fit(t, mags, dy, filts)
    P_multi = LS_multi.periodogram(periods)
    axi.plot(periods, P_multi, lw=1)
    
    text = ('$N_{{base}}={0},\ N_{{band}}={1}\ \ (M^{{eff}}={2})$'
            ''.format(Nbase, Nband, (2 * max(0, Nbase - Nband)
                                     + 5 * (2 * Nband + 1))))
    if (Nbase, Nband) == (1, 0):
        text += '    "shared-phase model"'
    elif (Nbase, Nband) == (0, 1):
        text += '    "multi-phase model"'

    axi.text(0.21, 0.98, text, fontsize=10, ha='left', va='top')

    axi.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    axi.yaxis.set_major_formatter(plt.NullFormatter())

ax[0].set_title('Periodograms for Multiterm Models')
ax[-1].set_xlabel('Period (days)')
ax[2].set_ylabel('multiterm model power')

fig.savefig('fig06.pdf')

plt.show()
