"""
This is the code used to compute various things used in the paper.
"""
import numpy as np
import os
from datetime import datetime

from mapcache import NumpyCache, compute_parallel
from gatspy.datasets import fetch_rrlyrae
from gatspy.periodic import (LombScargleMultiband, SuperSmoother,
                             SuperSmootherMultiband)


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


def compute_and_save_periods(rrlyrae, Model, outfile,
                             model_args=None, model_kwds=None,
                             Nperiods=5, save_every=5,
                             parallel=True, client=None,
                             num_results=None):
    """Function to compute periods and save the results"""
    cache = NumpyCache(outfile)
    lcids = rrlyrae.ids
    if num_results is not None:
        lcids = lcids[:num_results]

    # Define a function which, given an lcid, computes the desired periods.
    def find_periods(lcid, Nperiods=Nperiods,
                     rrlyrae=rrlyrae, Model=Model,
                     model_args=model_args, model_kwds=model_kwds):
        t, y, dy, filts = rrlyrae.get_lightcurve(lcid)
        model = Model(*(model_args or ()), **(model_kwds or {}))
        model.optimizer.period_range = (0.2, 1.2)
        model.optimizer.verbose = 0
        model.fit(t, y, dy, filts)
        return lcid, model.find_best_periods(Nperiods)

    return compute_parallel(cache, find_periods, lcids,
                            save_every=save_every,
                            parallel=parallel, client=client)
        
            
if __name__ == '__main__':
    rrlyrae = fetch_rrlyrae()
    rrlyrae_partial = fetch_rrlyrae(partial=True)

    parallel = True
    ssm_bands = ['g']

    if parallel:
        # Need some imports on the engine
        from IPython.parallel import Client
        client = Client()
        dview = client.direct_view()
        with dview.sync_imports():
            from gatspy.periodic import (LombScargleMultiband, SuperSmoother,
                                         SuperSmootherMultiband)

        # Make sure necessary data is fetched on all clients.
        # Otherwise, there can be cross-talk which results in an error.
        for c in client:
            c.block = True
            c.execute('from gatspy.datasets import fetch_rrlyrae;'
                      'fetch_rrlyrae()')
    else:
        client = None

    # Now time to compute the results. Here are the keywords used throughout:
    kwargs = dict(Nperiods=5, save_every=4, parallel=parallel, client=client)

    # Full Dataset
    compute_and_save_periods(rrlyrae, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=1, Nterms_band=0),
                             outfile='res_multiband_1_0', **kwargs)
    compute_and_save_periods(rrlyrae, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=0, Nterms_band=1),
                             outfile='res_multiband_0_1', **kwargs)
    for band in ssm_bands:
        compute_and_save_periods(rrlyrae, SuperSmoother1Band,
                                 model_kwds=dict(band=band),
                                 outfile='res_supersmoother_{0}'.format(band),
                                 **kwargs)

    # Partial Dataset
    compute_and_save_periods(rrlyrae_partial, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=1, Nterms_band=0),
                             outfile='res_partial_multiband_1_0', **kwargs)
    compute_and_save_periods(rrlyrae_partial, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=0, Nterms_band=1),
                             outfile='res_partial_multiband_0_1', **kwargs)
    for band in ssm_bands:
        compute_and_save_periods(rrlyrae_partial, SuperSmoother1Band,
                                 model_kwds=dict(band=band),
                                 outfile=('res_partial_supersmoother_'
                                          '{0}'.format(band)), **kwargs)
