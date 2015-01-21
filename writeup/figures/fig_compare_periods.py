"""
Plot figures comparing periods between multiband lomb scargle,
supersmoother, and Sesar 2010.

Note: results used here are computed using the ``compute_results.py`` script
in this directory.
"""
from __future__ import print_function, division

import os, sys; sys.path.append(os.path.abspath('../..'))

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

from multiband_LS import LombScargleMultiband, SuperSmootherMultiband
from compute_results import SuperSmoother1Band
from multiband_LS.data import fetch_light_curves

from compute_results import get_period_results



def plot_period_comparison(ax, Px_all, Py,
                           beats=[-3, -2, -1, 0, 1, 2, 3],
                           aliases=[], color=None):
    Px = Px_all[:, 0]

    ax.plot(Px, Py, 'o', alpha=0.5, ms=5, color=color)
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
        ax.plot(P1, fn(P1), '--', color='gray', alpha=0.7, lw=1, zorder=1)
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


def plot_example_lightcurve(rrlyrae, lcid):
    fig = plt.figure(figsize=(10, 4))

    gs = plt.GridSpec(2, 2,
                      left=0.07, right=0.95, wspace=0.1,
                      bottom=0.15, top=0.9)
    ax = [fig.add_subplot(gs[:, 0]),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 1])]

    t, y, dy, filts = rrlyrae.get_lightcurve(lcid, True)

    # don't plot data with huge errorbars
    mask = (dy < 1)
    t, y, dy, filts = t[mask], y[mask], dy[mask], filts[mask]
    period = rrlyrae.get_metadata(lcid)['P']
    phase = (t % period) / period

    for band in 'ugriz':
        mask = (filts == band)
        ax[0].errorbar(phase[mask], y[mask], dy[mask],
                       fmt='.', label=band)
    ax[0].legend(loc='upper left', ncol=3)
    ax[0].set(xlabel='phase', ylabel='magnitude',
              title='Folded Data (P={0:.3f} days)'.format(period))
    ylim = ax[0].get_ylim()
    ax[0].set_ylim(ylim[1], ylim[0] - 0.2 * (ylim[1] - ylim[0]))

    periods = np.linspace(0.2, 1.0, 4000)

    models = [SuperSmoother1Band(band='g'),
              LombScargleMultiband(Nterms_base=1, Nterms_band=0)]

    colors = seaborn.color_palette()

    for axi, model, color in zip(ax[1:], models, colors):
        model.fit(t, y, dy, filts)
        axi.plot(periods, model.score(periods), lw=1, color=color)
        axi.set_ylim(0, 1)

    ax[1].xaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_title("SuperSmoother on g-band")
    ax[2].set_title("Shared-phase Multiband")

    return fig, ax
    


def plot_periods(ssm_file, mbls_file, rrlyrae):
    ids = list(rrlyrae.ids)
    sesar_periods = np.array([rrlyrae.get_metadata(lcid)['P']
                              for lcid in ids])
    ssm_periods = get_period_results(ssm_file, ids)
    mbls_periods = get_period_results(mbls_file, ids)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.95, wspace=0.1,
                        bottom=0.15, top=0.9)

    colors = seaborn.color_palette()

    plot_period_comparison(ax[0], ssm_periods, sesar_periods, aliases=[1/2],
                           color=colors[0])
    plot_period_comparison(ax[1], mbls_periods, sesar_periods,
                           color=colors[1])

    ax[0].set_xlabel('supersmoother period (days)')
    ax[0].set_ylabel('Sesar 2010 period (days)')
    ax[1].set_xlabel('multiband period (days)')

    ax[0].set_title("SuperSmoother (g-band)", y=1.04)
    ax[1].set_title("Shared-phase Multiband", y=1.04)

    return fig, ax


if __name__ == '__main__':
    rrlyrae = fetch_light_curves()
    lcid = list(rrlyrae.ids)[4]

    fig, ax = plot_example_lightcurve(rrlyrae, lcid)
    fig.savefig('fig07a.pdf')

    fig, ax = plot_periods(ssm_file='results/supersmoother_g.npy',
                           mbls_file='results/multiband_1_0.npy',
                           rrlyrae=rrlyrae)
    fig.savefig('fig07b.pdf')
    plt.show()
    
