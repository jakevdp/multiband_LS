"""
Here we plot a typical approach to the multi-band periodogram: treating each
band separately, and taking a majority vote between the bands.
"""
import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()

from multiband_LS.generated import RRLyraeObject
from multiband_LS import (LombScargleAstroML, LombScargleMultiband,
                          NaiveMultiband)


# Choose a Sesar 2010 object to base our fits on
lcid = 1019544
rrlyrae = RRLyraeObject(lcid, random_state=0)
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
filts = np.array([f for f in 'ugriz'])

#----------------------------------------------------------------------
# First figure:
# Compute the lomb-scargle periodogram in each band

periods = np.linspace(0.2, 0.9, 1000)
model = NaiveMultiband(BaseModel=LombScargleAstroML)
model.fit(t, mags, dy, filts[:, None])
P = model.scores(periods)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1, bottom=0.15)

for i, band in enumerate('ugriz'):
    ax[0].errorbar((t / rrlyrae.period) % 1, mags[i], dy,
                   fmt='.', label=band)
ax[0].set_ylim(18, 14.5)
ax[0].legend(loc='upper left', fontsize=12, ncol=3)
ax[0].set_title('Folded Data, 5 bands per night (P={0:.3f} days)'.format(rrlyrae.period))
ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')

for i, band in enumerate(filts):
    offset = 4 - i
    ax[1].plot(periods, P[band] + offset, lw=1)
    ax[1].text(0.89, 0.7 + offset, band, fontsize=14, ha='right', va='top')
ax[1].set_title('Standard Periodogram in Each Band')
ax[1].set_ylim(0, 5)
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].set_xlabel('Period (days)')
ax[1].set_ylabel('power + offset')

fig.savefig('fig05a.pdf')

#----------------------------------------------------------------------
# Second figure:
# One observation per pass
    
# Alternate between the five bands. Because the times are randomized,
# the filter orders will also be randomized.
filts = np.take(list('ugriz'), np.arange(Nobs), mode='wrap')
mags = mags[np.arange(Nobs) % 5, np.arange(Nobs)]
    
masks = [(filts == band) for band in 'ugriz']

model = NaiveMultiband(BaseModel=LombScargleAstroML).fit(t, mags, dy, filts)
P = model.scores(periods)

fig = plt.figure(figsize=(10, 4))
gs = plt.GridSpec(5, 2,
                  left=0.07, right=0.95, bottom=0.15,
                  wspace=0.1, hspace=0.6)
ax = [fig.add_subplot(gs[:, 0]),
      fig.add_subplot(gs[:-2, 1]),
      fig.add_subplot(gs[-2:, 1])]
    
for band, mask in zip('ugriz', masks):
    ax[0].errorbar((t[mask] / rrlyrae.period) % 1, mags[mask], dy[mask],
                   fmt='.', label=band)
ax[0].set_ylim(18, 14.5)
ax[0].legend(loc='upper left', fontsize=12, ncol=3)
ax[0].set_title('Folded Data, 1 band per night (P={0:.3f} days)'.format(rrlyrae.period))
ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')

for i, band in enumerate('ugriz'):
    offset = 4 - i
    ax[1].plot(periods, P[band] + offset, lw=1)
    ax[1].text(0.89, 1 + offset, band,
               fontsize=10, ha='right', va='top')
ax[1].set_title('Standard Periodogram in Each Band')
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].xaxis.set_major_formatter(plt.NullFormatter())
ax[1].set_ylabel('power + offset')

for (i, Nbase, Nband) in [(0, 1, 0)]:#, (1, 0, 1)]:
    LS_multi = LombScargleMultiband(Nterms_base=Nbase, Nterms_band=Nband)
    LS_multi.fit(t, mags, dy, filts)
    P_multi = LS_multi.periodogram(periods)
    ax[2].plot(periods, P_multi + i, lw=1, color='gray')
    ax[2].text(0.89, 1.0 + i,
               "Nbase={0}, Nband={1}".format(Nbase, Nband),
               fontsize=10, ha='right', va='top')

ax[2].set_title('Multiband Periodogram')
ax[2].set_yticks([0, 0.5, 1.0])
ax[2].set_ylim(0, 1.0)
ax[2].yaxis.set_major_formatter(plt.NullFormatter())
ax[2].set_xlabel('Period (days)')
ax[2].set_ylabel('power')

fig.savefig('fig05b.pdf')

#----------------------------------------------------------------------
# Write the results to file
import pandas as pd
df = pd.DataFrame({'t': t})
df['band'] = filts
df['mag'] = mags
df['dmag'] = dy
df = df.sort('t')

df.to_csv('fig_data.dat', sep=' ', float_format='%.4f', index=False)

plt.show()
