import os
from datetime import datetime

import numpy as np

from mapcache import NumpyCache, compute_parallel
from LSSTsims import LSSTsims
from gatspy.periodic import (LombScargleMultiband,
                             SuperSmootherMultiband,
                             LombScargleMultibandFast)


class SuperSmoother1Band(SuperSmootherMultiband):
    """
    Convenience class to fit a single band of data with supersmoother

    This class ignores all data not associated with the given band.

    The main reason for this is that it can then be used as a stand-in for
    any multiband class.
    """
    def __init__(self, optimizer=None, band='g'):
        self.band = band
        SuperSmootherMultiband.__init__(self, optimizer)

    def _fit(self, t, y, dy, filts):
        import numpy as np
        mask = (filts == self.band)
        self.t, self.y, self.dy, self.filts = (t[mask], y[mask],
                                               dy[mask], filts[mask])
        self.unique_filts_ = np.unique(self.filts)
        return SuperSmootherMultiband._fit(self, self.t, self.y,
                                           self.dy, self.filts)


def compute_and_save_periods(Model, outfile,
                             pointing_indices, ndays,
                             rmags, template_indices,
                             model_args=None, model_kwds=None,
                             Nperiods=5, save_every=5,
                             parallel=True, client=None,
                             num_results=None):
    """Function to compute periods and save the results"""
    cache = NumpyCache(outfile)
    keys = list(np.broadcast(pointing_indices, ndays, rmags, template_indices))

    if num_results is not None:
        keys = keys[:num_results]

    # Define a function which, given a key, computes the desired periods.
    def find_periods(key, Nperiods=Nperiods, Model=Model, LSSTsims=LSSTsims,
                     model_args=model_args, model_kwds=model_kwds):
        import numpy as np
        lsstsim = LSSTsims()
        t, y, dy, filts = lsstsim.generate_lc(*key, random_state=0)

        model = Model(*(model_args or ()), **(model_kwds or {}))
        model.optimizer.period_range = (0.2, 1.2)
        model.optimizer.verbose = 0
        model.fit(t, y, dy, filts)

        try:
            periods = model.find_best_periods(Nperiods)
        except np.linalg.LinAlgError:
            periods = np.nan + np.zeros(Nperiods)
        except ValueError:
            periods = np.nan + np.zeros(Nperiods)
        return key, periods

    results = compute_parallel(cache, find_periods, keys,
                               save_every=save_every,
                               parallel=parallel, client=client)

    if num_results is not None:
        return results
    else:
        return gather_results(outfile, pointing_indices, ndays,
                              rmags, template_indices)


def gather_results(outfile, pointing_indices, ndays, rmags, template_indices):
    results = NumpyCache(outfile)
    brd = np.broadcast(pointing_indices, ndays, rmags, template_indices)
    results = np.array([results.get_row(key) for key in brd])
    return results.reshape(brd.shape + results.shape[-1:])


if __name__ == '__main__':
    parallel = True

    if parallel:
        # Need some imports on the engine
        from IPython.parallel import Client
        client = Client()
        dview = client.direct_view()
        with dview.sync_imports():
            from gatspy.periodic import (LombScargleMultiband,
                                         LombScargleMultibandFast,
                                         SuperSmootherMultiband)
    else:
        client = None

    template_indices = np.arange(2 * 23).reshape(2, 23).T
    pointing_indices = np.arange(1, 24)[:, None]
    ndays = np.array([90, 180, 365, 2*365])[:, None, None]
    rmags = np.array([20, 22, 24.5])[:, None, None, None]

    kwargs = dict(pointing_indices=pointing_indices,
                  ndays=ndays,
                  rmags=rmags,
                  template_indices=template_indices,
                  parallel=parallel, client=client,
                  save_every=4,
                  num_results=22)

    compute_and_save_periods(LombScargleMultiband, 'resultsLSST.npy',
                             model_kwds=dict(Nterms_base=1, Nterms_band=0),
                             **kwargs)

    compute_and_save_periods(LombScargleMultiband, 'resultsLSST01.npy',
                             model_kwds=dict(Nterms_base=0, Nterms_band=1),
                             **kwargs)

    for i, band in enumerate('ugrizy'):
        filename = 'resultsLSST_ssm_{0}.npy'.format(band)
        compute_and_save_periods(SuperSmoother1Band, filename,
                                 model_kwds=dict(band=i),
                                 **kwargs)
