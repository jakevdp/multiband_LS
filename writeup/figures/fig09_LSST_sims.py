from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

from gatspy.datasets import fetch_rrlyrae

import os, sys
sys.path.insert(0, 'LSSTsims')
from compute_results import gather_results


def olusei_period_criterion(Pobs, Ptrue, Tdays, dphimax = 0.037):
    factor = dphimax / Tdays
    return abs(Pobs - Ptrue) <= (Ptrue ** 2) * factor


template_indices = np.arange(2 * 23).reshape(2, 23).T
pointing_indices = np.arange(1, 24)[:, None]
ndays = np.array([180, 365, 2*365, 5*365])[:, None, None]
gmags = np.array([20, 21, 22, 23, 24.5])[:, None, None, None]

results_multi = 'LSSTsims/resultsLSST.npy'
results_ssm = 'LSSTsims/resultsLSST_ssm_{0}.npy'

# Get measured periods
Pobs_multi = gather_results(results_multi,
                            pointing_indices=pointing_indices,
                            ndays=ndays,
                            gmags=gmags,
                            template_indices=template_indices)
Pobs_multi[np.isnan(Pobs_multi)] = 0

Pobs_ssm = np.array([gather_results(results_ssm.format(band),
                                    pointing_indices=pointing_indices,
                                    ndays=ndays,
                                    gmags=gmags,
                                    template_indices=template_indices)
                     for band in 'ugriz'])
Pobs_ssm = Pobs_ssm[:, :, :, :, :, 0].transpose(1, 2, 3, 4, 0)
Pobs_ssm[np.isnan(Pobs_ssm)] = 0

# Get true periods
rrlyrae = fetch_rrlyrae()
Ptrue = np.reshape([rrlyrae.get_metadata(rrlyrae.ids[i])['P']
                    for i in template_indices.ravel()],
                   template_indices.shape)

# Check for matches
dphimax = 0.37
matches_multi = olusei_period_criterion(Pobs_multi,
                                        Ptrue.reshape(Ptrue.shape + (1,)),
                                        ndays.reshape(ndays.shape + (1,)),
                                        dphimax=dphimax)
results_multi = np.any(matches_multi, -1).mean(-1).mean(-1)

matches_ssm = olusei_period_criterion(Pobs_ssm,
                                      Ptrue.reshape(Ptrue.shape + (1,)),
                                      ndays.reshape(ndays.shape + (1,)),
                                      dphimax=dphimax)
results_ssm = np.any(matches_ssm, -1).mean(-1).mean(-1)

fig, ax = plt.subplots()
for t, frac_multi, frac_ssm in reversed(list(zip(ndays.ravel(),
                                                 results_multi.T,
                                                 results_ssm.T))):
    line = ax.plot(gmags.ravel(), frac_multi,
                   label='{0:.1f} years'.format(t / 365))
    line = ax.plot(gmags.ravel(), frac_ssm,
                   color=line[0].get_color(), linestyle='dashed')
    ax.fill_between(gmags.ravel(), frac_ssm, frac_multi,
                    edgecolor='none',
                    facecolor=line[0].get_color(), alpha=0.2)

ax.add_artist(ax.legend(loc='lower left'))
ref_lines = ax.plot([0], [0], '-k') + ax.plot([0], [0], '--k')
ax.legend(ref_lines, ['multiband', 'supersmoother'], loc='lower center')

ax.set(xlabel='g-band magnitude',
       ylabel='Fraction of Periods among Top-5',
       title='Multiband Improvement over SuperSmoother for LSST',
       xlim=(20, 24.5), ylim=(0, 1.05))

fig.savefig('fig09.pdf')
plt.show()
