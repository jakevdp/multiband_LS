"""
Here we plot a typical approach to the basic periodogram
"""

import sys
import os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()

from gatspy.datasets import RRLyraeGenerated
from gatspy.periodic import LombScargle


# Choose a Sesar 2010 object to base our fits on
lcid = 1019544
rrlyrae = RRLyraeGenerated(lcid, random_state=0)

# Generate data in a 6-month observing season
Nobs = 60
rng = np.random.RandomState(0)

nights = np.arange(180)
rng.shuffle(nights)
nights = nights[:Nobs]

t = 57000 + nights + 0.05 * rng.randn(Nobs)
dmag = 0.06 + 0.01 * rng.randn(Nobs)
mag = rrlyrae.generated('r', t, err=dmag, corrected=False)

periods = np.linspace(0.2, 1.4, 1000)

phase = (t / rrlyrae.period) % 1
phasefit = np.linspace(0, 1, 1000)
tfit = rrlyrae.period * phasefit

fig = plt.figure(figsize=(10, 4))
gs = plt.GridSpec(2, 2, left=0.07, right=0.95,
                  wspace=0.15, hspace=0.7,
                  bottom=0.15)
ax = [fig.add_subplot(gs[:, 0]),
      fig.add_subplot(gs[1, 1]),
      fig.add_subplot(gs[0, 1])]

# Plot the data
ax[0].errorbar(t, mag, dmag, fmt='o', color='#333333')
ax[1].errorbar(phase, mag, dmag, fmt='.', color='#888888')

# Fit and plot the model
model = LombScargle().fit(t, mag, dmag)
model.optimizer.period_range = (0.2, 1.2)

phase = (t / model.best_period) % 1
phasefit = np.linspace(0, 1, 1000)
tfit = model.best_period * phasefit
        
lines = ax[2].plot(periods, model.periodogram(periods), lw=1)
ax[1].plot(phasefit, model.predict(tfit),
           c=lines[0].get_color())
ax[1].invert_yaxis()

ax[0].set_xlabel('date of observation (MJD)')
ax[0].set_ylabel('magnitude')
ax[0].set_title('Input Data')
ax[0].invert_yaxis()

ax[1].set_title('Folded Data (P={0:.3f} days)'.format(rrlyrae.period))
ax[1].set_xlabel('phase')
ax[1].yaxis.set_major_locator(plt.MultipleLocator(0.2))

ax[2].set_title('Periodogram')
ax[2].set_xlabel('period (days)')
ax[2].set_ylabel('power')
ax[2].set_xlim(0.2, 1.4)
ax[2].set_ylim(0, 1)

plt.savefig('fig01.pdf')

plt.show()
