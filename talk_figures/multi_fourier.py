"""
Here we plot a typical approach to multi-band Lomb-Scargle: treating each band
separately, and taking a majority vote between the bands.
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


for fignum, models in enumerate(([1, 2, 6], [10, 20, 30])):
    fig = plt.figure(figsize=(10, 4))
    gs = plt.GridSpec(3, 2, left=0.07, right=0.95, bottom=0.15,
                      wspace=0.15, hspace=0.3)

    ax = [fig.add_subplot(gs[:, 0]),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 1]),
          fig.add_subplot(gs[2, 1])]

    # Plot the data
    ax[0].errorbar(phase, mag, dmag, fmt='o', color='#AAAAAA')

    # Plot the fits
    for i, Nterms in enumerate(models):
        model = LombScargle(Nterms=Nterms).fit(t, mag, dmag)
        P = model.periodogram(periods)

        label = "{0} terms".format(Nterms)
        if Nterms == 1:
            label = label[:-1]

        alpha = 1.0 if (Nterms < 20) else 0.5

        lines = ax[0].plot(phasefit, model.predict(tfit, period=rrlyrae.period),
                           label=label, alpha=alpha)

        ax[1 + i].plot(periods, model.periodogram(periods),
                       color=lines[0].get_color(), lw=1)
        ax[1 + i].set_title("{0}-term Periodogram".format(Nterms))
        ax[1 + i].set_xlim(0.2, 1.4)
        ax[1 + i].set_ylim(0, 1)

    ax[2].set_ylabel('power')
    ax[3].set_xlabel('period (days)')
    for i in [1, 2]:
        ax[i].xaxis.set_major_formatter(plt.NullFormatter())

    ax[0].set_xlabel('phase')
    ax[0].set_ylabel('magnitude')
    ax[0].invert_yaxis()
    ax[0].legend(loc='upper left')
    ax[0].set_title('Folded Data (P={0:.3f} days)'.format(rrlyrae.period))
    ax[0].set_ylim(16.4, 15.5)

    fig.savefig('multiterm_{0}.png'.format(fignum + 1))
           
plt.show()
