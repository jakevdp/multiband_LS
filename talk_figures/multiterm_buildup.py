import numpy as np
import matplotlib.pyplot as plt

# Use seaborn settings for plot styles
import seaborn; seaborn.set()

from gatspy.datasets import RRLyraeGenerated
from gatspy.periodic import LombScargle, LombScargleMultiband


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
filts = np.array([f for f in 'ugriz'])

def plot_data(ax):
    for i, band in enumerate('ugriz'):
        ax.errorbar((t / rrlyrae.period) % 1, mags[i], dy,
                    fmt='.', label=band)
    ax.set_ylim(18, 14.5)
    ax.legend(loc='upper left', fontsize=12, ncol=3)
    ax.set_xlabel('phase')
    ax.set_ylabel('magnitude')

# Plot the input data
fig, ax = plt.subplots()
plot_data(ax)
ax.set_title('Input Data')
plt.savefig('buildup_1.png')

# Plot the base model
fig, ax = plt.subplots()
plot_data(ax)

t_all = np.ravel(t * np.ones_like(mags))
mags_all = np.ravel(mags)
dy_all = np.ravel(dy * np.ones_like(mags))
basemodel = LombScargle(Nterms=2).fit(t_all, mags_all, dy_all)

period = rrlyrae.period
tfit = np.linspace(0, period, 1000)
base_fit = basemodel.predict(tfit, period=period)

ax.plot(tfit / period, base_fit, color='black', lw=5, alpha=0.5)
ax.set_title('2-term Base Model')

# Plot the band-by-band augmentation
multimodel = LombScargleMultiband(Nterms_base=2, Nterms_band=1)
t1, y1, dy1, f1 = map(np.ravel,
                      np.broadcast_arrays(t, mags, dy, filts[:, None]))
multimodel.fit(t1, y1, dy1, f1)


yfits = multimodel.predict(tfit, filts=filts[:, None], period=period)
plt.savefig('buildup_2.png')

fig, ax = plt.subplots()
for i in range(5):
    ax.plot(tfit / period, yfits[i] - base_fit)

ax.plot(tfit / period, 0 * tfit, '--k')
ax.set_ylim(1.7, -1.8)
ax.set_xlabel('phase')
ax.set_ylabel('magnitude')
ax.set_title('1-term Band offset')
plt.savefig('buildup_3.png')


# Plot the final model
fig, ax = plt.subplots()

plot_data(ax)
ax.plot(tfit / period, base_fit, color='black', lw=10, alpha=0.2)

for i in range(5):
    ax.plot(tfit / period, yfits[i])

ax.set_title('Final Model')
plt.savefig('buildup_4.png')


plt.show()
