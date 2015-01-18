"""
Tools for computing periods with various methods
"""
import os, sys; sys.path.append(os.path.abspath('../..'))

import contextlib
import numpy
import os
from IPython import parallel

import multiband_LS
from multiband_LS.data import fetch_light_curves
from multiband_LS.memoize import CacheResults


DIRNAME = os.path.dirname(os.path.abspath(__file__))
print("ComputePeriods home directory: {0}".format(DIRNAME))


def best_period_Multiband(lcid, rrlyrae,
                          Nterms_base=1, Nterms_band=0):
    """Best period using the Multi-band Lomb-Scargle algorithm.

    Compute the best period for the given light curve, using the multiband
    model with the given Nbase and Nband
    """
    t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
    ls = multiband_LS.LombScargleMultiband(Nterms_base=Nterms_base,
                                           Nterms_band=Nterms_band)
    ls.optimizer.period_range = (0.2, 1.2)
    ls.fit(t, y, dy, filts)
    return ls.find_best_periods(5)


def periods_Multiband(lcids=None, Nterms_base=1, Nterms_band=0,
                      func=best_period_Multiband, dirname=DIRNAME):
    """Compute and cache best periods for the given light curve ids"""
    cachedir = 'results_multiband_{0}_{1}'.format(Nterms_base, Nterms_band)
    cachedir = os.path.join(dirname, cachedir)
    cache = CacheResults(cachedir, verbose=True)
    rrlyrae = fetch_light_curves()
    if lcids is None:
        lcids = rrlyrae.ids
    results = cache.call_iter(func, lcids,
                              args=(rrlyrae, Nterms_base, Nterms_band))
    return numpy.asarray(results)


def best_period_SuperSmoother(lcid, rrlyrae, filt='g'):
    """Best period using the SuperSmoother algorithm

    Compute the best period for the given light curve, using data from the
    given filter.
    """
    t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
    t, y, dy = (x[filts == filt] for x in (t, y, dy))
    ssm = multiband_LS.SuperSmoother()
    ssm.optimizer.period_range = (0.2, 1.2)
    ssm.fit(t, y, dy)
    return ssm.find_best_periods(5)


def periods_SuperSmoother(lcids=None, filt='g',
                          func=best_period_SuperSmoother, dirname=DIRNAME):
    """Compute and cache best periods for the given light curve ids"""
    cachedir = 'results_supersmoother_{0}'.format(filt)
    cachedir = os.path.join(dirname, cachedir)
    cache = CacheResults(cachedir, verbose=True)
    rrlyrae = fetch_light_curves()
    if lcids is None:
        lcids = rrlyrae.ids
    results = cache.call_iter(func, lcids, args=(rrlyrae, filt))
    return numpy.asarray(results)


def parallelize(func, *client_args, **client_kwargs):
    """Parallelize one of the above functions"""

    @contextlib.wraps(func)
    def wrapper(lcids, *args, **kwargs):
        try:
            client = parallel.Client(*client_args, **client_kwargs)
        except FileNotFoundError:
            raise ValueError("Error accessing IPython Parallel clients. "
                             "Did you run ``ipcluster start``?")
            
        print("client ids:", client.ids)

        dview = client.direct_view()
        with dview.sync_imports():
            import os
            import numpy
            import multiband_LS
            from multiband_LS.memoize import CacheResults
            from multiband_LS.data import fetch_light_curves

        # Make sure the light curves are fetched before parallelizing,
        # otherwise this may lead to parallel file downloads!
        client[0].execute('fetch_light_curves()')

        lbv = client.load_balanced_view()
        lbv.block = False

        lcid_batches = [[lcid] for lcid in lcids]

        results = lbv.map(func, lcid_batches, args=args, kwargs=kwargs)

        from time import time
        t0 = time()
        for i, result in enumerate(results):
            print("{0}/{1} : {2}".format(i + 1, len(results), result))
            print("     elapsed: {0:.0f} sec".format(time() - t0))
        return numpy.concatenate(results)
    return wrapper


periods_Multiband_parallel = parallelize(periods_Multiband)
periods_SuperSmoother_parallel = parallelize(periods_SuperSmoother)


if __name__ == '__main__':
    rrlyrae = fetch_light_curves()
    ids = list(rrlyrae.ids)
    print(periods_Multiband_parallel(ids))
    print(periods_SuperSmoother_parallel(ids))
