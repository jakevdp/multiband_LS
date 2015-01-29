"""
This is the code used to compute various things used in the paper.
"""
import numpy as np
import os
from datetime import datetime

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
    # Here is the datatype of the results array:
    dtype = [('id', 'int32'), ('periods', '{0:d}float64'.format(Nperiods))]

    if outfile[-4:] != '.npy':
        outfile += '.npy'

    if os.path.exists(outfile):
        prev_results = np.load(outfile)
        if prev_results.dtype != dtype:
            raise ValueError("dtype of previous results does not match.")
        print("using {0} previous results from {1}"
              "".format(len(prev_results['id']), outfile))
        lcids = [lcid for lcid in rrlyrae.ids
                 if lcid not in prev_results['id']]
    else:
        prev_results = np.zeros(0, dtype=dtype)
        lcids = list(rrlyrae.ids)

    # For testing purposes... only find a few results
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

    # Set up the iterator over results
    results = np.zeros(len(lcids), dtype=dtype)
    if parallel:
        if client is None:
            from IPython.parallel import Client
            client = Client()
        lbv = client.load_balanced_view()
        results_iter = enumerate(lbv.map(find_periods, lcids,
                                         block=False, ordered=False))
    else:
        results_iter = enumerate(map(find_periods, lcids))

    # Do the iteration, saving the results occasionally
    print(datetime.now())
    for i, (lcid, periods) in results_iter:
        print('{0}/{1}: {2} {3}'.format(i + 1, len(lcids), lcid, periods))
        print(' {0}'.format(datetime.now()))
        results['id'][i] = lcid
        results['periods'][i] = periods

        if (i + 1) % save_every == 0:
            np.save(outfile, np.concatenate([prev_results,
                                             results[:i + 1]]))

    results = np.concatenate([prev_results, results])
    np.save(outfile, results)
    return results


def get_period_results(outfile, lcids=None):
    """
    Get the results from the outfile for the associated integer light curve ids
    """
    try:
        unordered_results = np.load(outfile)
    except FileNotFoundError:
        raise FileNotFoundError("File {0} not found. You can run the code in "
                                "compute_results.py to create the file."
                                "".format(outfile))

    if lcids is None:
        lcids = fetch_rrlyrae().ids

    # turn generator into list
    lcids = list(lcids)
    dtype = unordered_results.dtype

    D = dict(zip(unordered_results['id'], unordered_results['periods']))
    results = np.zeros(len(lcids), dtype=dtype)
    for i, lcid in enumerate(lcids):
        results['id'][i] = lcid
        results['periods'][i] = D[lcid]
    return results['periods']
        
            
if __name__ == '__main__':
    rrlyrae = fetch_rrlyrae()
    rrlyrae_partial = fetch_rrlyrae(partial=True)

    if not os.path.exists('results'):
        os.makedirs('results')

    # Need some imports on the engine
    from IPython.parallel import Client
    client = Client()
    dview = client.direct_view()
    with dview.sync_imports():
        from gatspy.periodic import (LombScargleMultiband, SuperSmoother,
                                     SuperSmootherMultiband)

    # Now time to compute the results. Here are the keywords used throughout:
    kwargs = dict(Nperiods=5, save_every=1, parallel=True, client=client)

    # Full Dataset
    compute_and_save_periods(rrlyrae, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=1, Nterms_band=0),
                             outfile='results/multiband_1_0', **kwargs)
    compute_and_save_periods(rrlyrae, SuperSmoother1Band,
                             outfile='results/supersmoother_g', **kwargs)
    #compute_and_save_periods(rrlyrae, SuperSmootherMultiband,
    #                         outfile='results/supersmoother_multi',
    #                         **kwargs)

    # Partial Dataset
    compute_and_save_periods(rrlyrae_partial, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=1, Nterms_band=0),
                             outfile='results/partial_multiband_1_0', **kwargs)
    compute_and_save_periods(rrlyrae_partial, SuperSmoother1Band,
                             outfile='results/partial_supersmoother_g',
                             **kwargs)
    #compute_and_save_periods(rrlyrae_partial, SuperSmootherMultiband,
    #                         outfile='results/partial_supersmoother_multi',
    #                         **kwargs)


    # Additional Multiterm Model
    compute_and_save_periods(rrlyrae, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=0, Nterms_band=1),
                             outfile='results/multiband_0_1', **kwargs)
    compute_and_save_periods(rrlyrae_partial, LombScargleMultiband,
                             model_kwds=dict(Nterms_base=0, Nterms_band=1),
                             outfile='results/partial_multiband_0_1', **kwargs)
