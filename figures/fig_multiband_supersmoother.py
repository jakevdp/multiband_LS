"""
Here we plot a typical approach to the multi-band periodogram: treating each
band separately, and taking a majority vote between the bands.
"""

import sys
import os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()

from gatspy.datasets import RRLyraeGenerated
from gatspy.periodic import SuperSmoother


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

t = 57000 + nights + 0.05 * rng.randn(Nobs)
dy = 0.06 + 0.01 * rng.randn(Nobs)
mags = np.array([rrlyrae.generated(band, t, err=dy, corrected=False)
                 for band in 'ugriz'])

periods = np.linspace(0.2, 0.9, 1000)

#----------------------------------------------------------------------
# First figure:
# Compute the supersmoother periodogram in each band

P = [SuperSmoother().fit(t, m, dy).periodogram(periods) for m in mags]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1, bottom=0.15)

for i, band in enumerate('ugriz'):
    ax[0].errorbar((t / rrlyrae.period) % 1, mags[i], dy,
                   fmt='.', label=band)
ax[0].set_ylim(18, 14.5)
ax[0].legend(loc='upper left', fontsize=12, ncol=3)
ax[0].set_title('Folded Data (P={0:.3f} days)'.format(rrlyrae.period))
ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')

for i, (Pi, band) in enumerate(zip(P, 'ugriz')):
    offset = 4 - i
    ax[1].plot(periods, Pi + offset, lw=1)
    ax[1].text(0.89, 0.7 + offset, band, fontsize=14, ha='right', va='top')
ax[1].set_title('Periodogram in Each Band')
ax[1].set_ylim(0, 5)
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].set_xlabel('Period (days)')
ax[1].set_ylabel('power + offset')

#----------------------------------------------------------------------
# Second figure:
# Use a truncated dataset
    
# Alternate between the five bands. Because the times are randomized,
# the filter orders will also be randomized.
filts = np.take(list('ugriz'), np.arange(Nobs), mode='wrap')
mags = mags[np.arange(Nobs) % 5, np.arange(Nobs)]
    
masks = [(filts == band) for band in 'ugriz']

P = [SuperSmoother().fit(t[mask], mags[mask],
                         dy[mask]).periodogram(periods)
     for mask in masks]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1, bottom=0.15)

for band, mask in zip('ugriz', masks):
    ax[0].errorbar(t[mask] / rrlyrae.period % 1, mags[mask], dy[mask],
                   fmt='.', label=band)
ax[0].set_ylim(18, 14.5)
ax[0].legend(loc='upper left', fontsize=12, ncol=3)
ax[0].set_title('Folded Data (P={0:.3f} days)'.format(rrlyrae.period))
ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')

for i, (Pi, band) in enumerate(zip(P, 'ugriz')):
    offset = 4 - i
    ax[1].plot(periods, Pi + offset, lw=1)
    ax[1].text(0.89, 0.7 + offset, band, fontsize=14, ha='right', va='top')
ax[1].set_title('Periodogram in Each Band')
ax[1].set_ylim(0, 5)
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].set_xlabel('Period (days)')
ax[1].set_ylabel('power + offset')


plt.show()
