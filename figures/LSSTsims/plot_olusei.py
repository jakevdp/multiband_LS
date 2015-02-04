import numpy as np
import matplotlib.pyplot as plt

from compute_results import gather_results
from gatspy.datasets import fetch_rrlyrae


def period_criterion(Pobs, Ptrue, Tdays, dphimax = 0.037):
    factor = dphimax / Tdays
    return abs(Pobs - Ptrue) <= (Ptrue ** 2) * factor


template_indices = np.arange(2 * 23).reshape(2, 23).T
pointing_indices = np.arange(1, 24)[:, None]
ndays = np.array([90, 180, 365, 2*365])[:, None, None]
gmags = np.array([20, 22, 24.5])[:, None, None, None]

rrlyrae = fetch_rrlyrae()
Ptrue = np.reshape([rrlyrae.get_metadata(rrlyrae.ids[i])['P']
                    for i in template_indices.ravel()],
                   template_indices.shape)


def period_recovery(band):
    outfile = 'resultsLSST_ssm_{0}.npy'.format(band)
    results = gather_results(outfile,
                             pointing_indices=pointing_indices,
                             ndays=ndays,
                             gmags=gmags,
                             template_indices=template_indices)
    best_results = results[:, :, :, :, 0]
    recovery = period_criterion(best_results, Ptrue, ndays)
    print(band, recovery.sum() / recovery.size)
    return recovery


matches = (np.sum([period_recovery(band) for band in 'gri'], 0) >= 2)
frac = matches.mean(-1).mean(-1)

print(matches.shape)
print(frac.shape)

for i, t in enumerate(ndays.ravel()):
    plt.plot(gmags.ravel(), frac[:, i], '-s', label='{0:.1f} yrs'.format(t / 365))
plt.legend()
plt.xlim(20, 27)
plt.ylim(0, 1)
plt.xlabel('g-band mag')
plt.ylabel('fraction of correct periods')

plt.show()
