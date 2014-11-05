import numpy as np
import matplotlib.pyplot as plt

from astroML.time_series import lomb_scargle
from multiband_LS.data import fetch_light_curves

from multiband_LS.period_search import period_search

rrlyrae = fetch_light_curves()
my_id = list(rrlyrae.ids)[300]
t, y, dy = rrlyrae.get_lightcurve(my_id)

periods = period_search(t, y, dy)
print(np.unique(periods))

best_period = np.median(periods)

for i, filt in enumerate('ugriz'):
    plt.errorbar(t[:, i] % best_period, y[:, i], dy[:, i],
                 fmt='.', ecolor='gray', label=filt)
plt.legend(loc='best', ncol=5, numpoints=1, frameon=False)
plt.ylim(plt.ylim()[0] - 0.5, None)
plt.gca().invert_yaxis()
plt.show()
