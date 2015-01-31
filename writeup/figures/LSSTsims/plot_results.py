import numpy as np
import matplotlib.pyplot as plt

from compute_results import gather_results
from gatspy.datasets import fetch_rrlyrae


def plot_results(outfile, pointing_indices, ndays, rmags, template_indices):
    results = gather_results(outfile,
                             pointing_indices=pointing_indices,
                             ndays=ndays,
                             rmags=rmags,
                             template_indices=template_indices)

    rrlyrae = fetch_rrlyrae()
    periods = np.reshape([rrlyrae.get_metadata(rrlyrae.ids[i])['P']
                          for i in template_indices.ravel()],
                         template_indices.shape)

    nans = np.isnan(results)
    results[nans] = 0

    matches = np.any(abs(results - periods[:, :, None]) < 0.01, -1)
    #matches[np.any(nans, -1)] = np.nan
    matches = matches.reshape(matches.shape[:2] + (-1,))

    frac = np.nanmean(matches, -1)

    rmags = rmags.ravel()
    ndays = ndays.ravel()

    for i in range(len(rmags)):
        plt.plot(ndays / 365, frac[i], label='r = {0:.2f}'.format(rmags[i]))
    plt.grid(True)
    plt.legend(loc='lower right')


def plot_results_multi(outfiles, pointing_indices, ndays, rmags, template_indices):
    results = [gather_results(outfile,
                              pointing_indices=pointing_indices,
                              ndays=ndays,
                              rmags=rmags,
                              template_indices=template_indices)[:, :, :, :, :1]
               for outfile in outfiles]
    results = np.concatenate(results, axis=-1)

    rrlyrae = fetch_rrlyrae()
    periods = np.reshape([rrlyrae.get_metadata(rrlyrae.ids[i])['P']
                          for i in template_indices.ravel()],
                         template_indices.shape)

    nans = np.isnan(results)
    results[nans] = 0

    matches = np.any(abs(results - periods[:, :, None]) < 0.01, -1)
    #matches[np.any(nans, -1)] = np.nan
    matches = matches.reshape(matches.shape[:2] + (-1,))

    frac = np.nanmean(matches, -1)

    rmags = rmags.ravel()
    ndays = ndays.ravel()

    for i in range(len(rmags)):
        plt.plot(ndays / 365, frac[i], label='r = {0:.2f}'.format(rmags[i]))
    plt.grid(True)




template_indices = np.arange(5 * 23).reshape(5, 23).T
pointing_indices = np.arange(1, 24)[:, None]
ndays = np.array([90, 180, 365, 2*365])[:, None, None]
rmags = np.array([20, 22, 24.5])[:, None, None, None]

plot_results('resultsLSST.npy',
             pointing_indices=pointing_indices,
             ndays=ndays,
             rmags=rmags,
             template_indices=template_indices)

template_indices = template_indices[:, :1]

plot_results_multi(['resultsLSST_ssm_{0}.npy'.format(band)
                    for band in 'g'],
                   pointing_indices=pointing_indices,
                   ndays=ndays,
                   rmags=rmags,
                   template_indices=template_indices)

plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.25))
plt.ylim(0, 1)
plt.xlabel('LSST observing time (years)')
plt.ylabel('fraction of periods recovered')

plt.show()
