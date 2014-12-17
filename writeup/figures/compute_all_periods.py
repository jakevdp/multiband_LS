import sys, os; sys.path.append(os.path.abspath('../..'))

from multiband_LS import LombScargleMultiband
from multiband_LS.memoize import CacheResults

from multiband_LS.data import fetch_light_curves
rrlyrae = fetch_light_curves()

def find_best_period(lcid, rrlyrae, Nterms_base=1, Nterms_band=0):
    t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
    LS = LombScargleMultiband(Nterms_base=Nterms_base,
                              Nterms_band=Nterms_band,
                              period_range=(0.2, 1.1))
    LS.fit(t, y, dy, filts)
    return LS.best_period


lcids = list(rrlyrae.ids)

cache = CacheResults('results_1_0', verbose=True)
periods_1_0 = cache.call_iter(find_best_period, lcids,
                              args=(rrlyrae, 1, 0))

cache = CacheResults('results_0_1', verbose=True)
periods_0_1 = cache.call_iter(find_best_period, lcids,
                              args=(rrlyrae, 0, 1))
