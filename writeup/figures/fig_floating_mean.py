"""
Illustration of a floating-mean periodogram
"""

import sys
import os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()

from multiband_LS.generated import RRLyraeObject
from multiband_LS.lomb_scargle import LombScargle


# Choose a Sesar 2010 object to base our fits on
lcid = 1019544
rrlyrae = RRLyraeObject(lcid, random_state=0)

# Generate data in a 6-month observing season
Nobs = 60
rng = np.random.RandomState(0)

nights = np.arange(180)
rng.shuffle(nights)
nights = nights[:Nobs]

t = 57000 + nights + 0.1 * rng.randn(Nobs)
dmag = 0.06 + 0.01 * rng.randn(Nobs)
mag = (16 + 0.5 * np.sin(2 * np.pi * t / rrlyrae.period - 0.4)
       + dmag * rng.randn(Nobs))
omega_best = 2 * np.pi / rrlyrae.period
phase = (t / rrlyrae.period) % 1

mask = mag < 16.05

phasefit = np.linspace(0, 1, 1000)
tfit = rrlyrae.period * phasefit

periods = np.linspace(0.2, 1.1, 1000)
omegas = 2 * np.pi / periods

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1, bottom=0.15)

ax[0].errorbar(phase[mask], mag[mask], dmag[mask], fmt='o',
               color='#666666')
ax[0].errorbar(phase[~mask], mag[~mask], dmag[~mask], fmt='o',
               color='#CCCCCC')
ax[0].invert_yaxis()

for fit_offset in [False, True]:
    model = LombScargle(fit_offset=fit_offset).fit(t[mask],
                                                   mag[mask], dmag[mask])
    P = model.periodogram(omegas)
    if fit_offset:
        label = 'floating mean'
    else:
        label = 'standard'
    ax[0].plot(phasefit, model.predict(tfit, omega=2 * np.pi / rrlyrae.period),
               label=label)
    ax[1].plot(periods, P + 1 - fit_offset, lw=1)
    ax[1].text(1.08, 0.5 + 1 - fit_offset, label, ha='right')

ax[0].legend(loc='upper left')
ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')
ax[0].set_title('Phased data (period=0.622 days)')

ax[1].set_xlabel('Period (days)')
ax[1].set_ylabel('power + offset')
ax[1].set_ylim(0, 2)
ax[1].set_xlim(0.2, 1.1)
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].set_title('Periodogram for each model')

plt.savefig('fig02.pdf')
plt.show()
