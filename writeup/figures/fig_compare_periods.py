"""
Plot figures comparing periods between multiband lomb scargle,
supersmoother, and Sesar 2010.
"""
from __future__ import print_function, division

import os, sys; sys.path.append(os.path.abspath('../..'))

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

from multiband_LS import (LombScargleAstroML, LombScargleMultiband,
                          SuperSmoother)
from multiband_LS.memoize import CacheResults
from multiband_LS.data import fetch_light_curves

from compute_results import get_period_results
from compute_rrlyrae_periods import periods_Multiband, periods_SuperSmoother



def plot_period_comparison(ax, Px_all, Py,
                           beats=[-3, -2, -1, 0, 1, 2, 3],
                           aliases=[]):
    Px = Px_all[:, 0]

    ax.plot(Px, Py, 'o', alpha=0.5, ms=5)
    P1 = np.linspace(0.1, 1.2)

    matches = lambda x, y, tol=0.01: np.sum(abs(x - y) < tol)

    for n in beats:
        fn = lambda P, n=n: P / (1 + abs(n) * P)

        if n == 0:
            ax.plot(P1, P1, ':k', alpha=0.7, lw=1, zorder=1)
            ax.text(1.21, 1.21, str(matches(Px, fn(Py))),
                    size=10, va='bottom', ha='left', color='gray')
        elif n > 0:
            ax.plot(P1, fn(P1), ':k', alpha=0.7, lw=1, zorder=1)
            ax.text(1.21, fn(1.2), str(matches(fn(Px), Py)),
                    size=10, va='center', ha='left', color='gray')
        else:
            ax.plot(fn(P1), P1, ':k', alpha=0.7, lw=1, zorder=1)
            ax.text(fn(1.2), 1.21, str(matches(Px, fn(Py))),
                    size=10, va='bottom', ha='center', color='gray')

    for n in aliases:
        fn = lambda P, n=n: n * P
        ax.plot(P1, fn(P1), ':k', alpha=0.7, lw=1, zorder=1)
        if n < 1:
            ax.text(1.21, fn(1.2), str(matches(fn(Px), Py)),
                    size=10, va='center', ha='left', color='gray')
        else:
            ax.text(fn(1.2), 1.21, str(matches(Px, fn(Py))),
                    size=10, va='bottom', ha='center', color='gray')

    ax.set_xlim(0.1, 1.2)
    ax.set_ylim(0.1, 1.2)


    top_matches = np.any((abs(Px_all.T - Py) < 0.01), 0).sum()
    ax.text(1.17, 0.11, "Top 5 Matches: {0}/{1}".format(top_matches, len(Px)),
            size=10, ha='right', va='bottom')


def plot_periods(ssm_file, mbls_file):
    rrlyrae = fetch_light_curves()

    ids = list(rrlyrae.ids)
    sesar_periods = np.array([rrlyrae.get_metadata(lcid)['P']
                              for lcid in ids])
    ssm_periods = get_period_results(ssm_file)
    mbls_periods = get_period_results(mbls_file)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1,
                        bottom=0.15, top=0.9)

    plot_period_comparison(ax[0], ssm_periods, sesar_periods, aliases=[1/2])
    plot_period_comparison(ax[1], mbls_periods, sesar_periods)

    ax[0].set_xlabel('supersmoother period (days)')
    ax[0].set_ylabel('Sesar 2010 period (days)')
    ax[1].set_xlabel('multiband period (days)')

    ax[0].set_title("SuperSmoother (single-band)", y=1.04)
    ax[1].set_title("Multiband (1, 0)-model", y=1.04)

    return fig, ax


if __name__ == '__main__':
    fig, ax = plot_periods(ssm_file='results/supersmoother_g.npy',
                           mbls_file='results/multiband_1_0.npy')
    fig.savefig('fig07.pdf')
    plt.show()
    
