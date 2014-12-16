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

t = 57000 + nights + 0.05 * rng.randn(Nobs)
dmag = 0.06 + 0.01 * rng.randn(Nobs)
mag = rrlyrae.generated('r', t, err=dmag, corrected=False)
omega_best = 2 * np.pi / rrlyrae.period

periods = np.linspace(0.2, 1.4, 1000)
omegas = 2 * np.pi / periods

phase = (t / rrlyrae.period) % 1
phasefit = np.linspace(0, 1, 1000)
tfit = rrlyrae.period * phasefit

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1, bottom=0.15)

# Plot the data
ax[0].errorbar(phase, mag, dmag, fmt='o', color='#AAAAAA')

# Plot the fits
models = [1, 2, 6]

for i, Nterms in enumerate(models):
    model = LombScargle(Nterms=Nterms).fit(t, mag, dmag)
    P = model.periodogram(omegas)

    label = "{0} terms".format(Nterms)
    if Nterms == 1:
        label = label[:-1]
        
    offset = len(models) - 1 - i
    lines = ax[1].plot(periods, offset + model.periodogram(omegas), lw=1)
    ax[1].text(0.22, offset + 0.9, label, ha='left', va='top')
    ax[0].plot(phasefit, model.predict(tfit, omega=omega_best),
               c=lines[0].get_color(), label=label)

ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')
ax[0].invert_yaxis()
ax[0].legend(loc='upper left')
ax[0].set_title('Folded Data (P={0:.3f} days)'.format(rrlyrae.period))

ax[1].set_title('Periodogram for each model')
ax[1].set_xlabel('period (days)')
ax[1].set_ylabel('power + offset')
ax[1].set_xlim(0.2, 1.4)
ax[1].set_ylim(0, len(models))
ax[1].yaxis.set_major_formatter(plt.NullFormatter())

plt.savefig('fig03.pdf')
           
plt.show()
