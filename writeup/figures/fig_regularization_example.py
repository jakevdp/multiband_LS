"""
Here we plot an example of how regularization can affect the fit
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

fig = plt.figure(figsize=(10, 4))
gs = plt.GridSpec(2, 2, left=0.07, right=0.95, wspace=0.15, bottom=0.15)
ax = [fig.add_subplot(gs[:, 0]),
      fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, 1])]

# Plot the data
ax[0].errorbar(phase, mag, dmag, fmt='o', color='#AAAAAA')
ylim = ax[0].get_ylim()

# Here we construct some regularization.
Nterms = 20
sigma_r_inv =  np.vstack([np.arange(Nterms + 1),
                          np.arange(Nterms + 1)]).T.ravel()[1:] ** 2
models = [0.5 * sigma_r_inv ** 2, None]

for i, reg in enumerate(models):
    model = LombScargle(Nterms=Nterms, regularization=reg,
                        regularize_by_trace=False).fit(t, mag, dmag)
    P = model.periodogram(omegas)

    if reg is None:
        label = "unregularized"
    else:
        label = "regularized"
        
    lines = ax[0].plot(phasefit, model.predict(tfit, omega=omega_best),
                       label=label)

    ax[1 + i].plot(periods, model.periodogram(omegas), lw=1,
                   c=lines[0].get_color())
    ax[1 + i].set_title("{0} Periodogram ({1} terms)".format(label.title(),
                                                             Nterms))
    ax[1 + i].set_ylabel('power')
    ax[1 + i].set_xlim(0.2, 1.4)
    ax[1 + i].set_ylim(0, 1)
    #ax[1 + i].yaxis.set_major_formatter(plt.NullFormatter())

ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')
ax[0].set_ylim(ylim)
ax[0].invert_yaxis()
ax[0].legend(loc='upper left')
ax[0].set_title('Folded Data (P={0:.3f} days)'.format(rrlyrae.period))

ax[1].xaxis.set_major_formatter(plt.NullFormatter())
ax[2].set_xlabel('period (days)')

plt.savefig('fig04.pdf')
           
plt.show()
