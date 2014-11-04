import numpy as np
import matplotlib.pyplot as plt

from astroML.time_series import lomb_scargle
from multiband_LS.data import fetch_light_curves

from multiband_LS.period_search import period_search

rrlyrae = fetch_light_curves()
my_id = rrlyrae.ids.next()
t, y, dy = rrlyrae.get_lightcurve(my_id)

periods = period_search(t, y, dy)
print(periods)

best_period = np.median(periods)

for i in range(5):
    plt.errorbar(t[:, i] % best_period, y[:, i], dy[:, i],
                 fmt='.', ecolor='gray')
plt.show()
