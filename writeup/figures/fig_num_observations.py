import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from multiband_LS.data import fetch_light_curves


rrlyrae = fetch_light_curves()
rrlyrae_partial = fetch_light_curves(partial=True)

data = [rrlyrae.get_lightcurve(lcid, True)
        for lcid in rrlyrae.ids]
data_partial = [rrlyrae_partial.get_lightcurve(lcid, True)
                for lcid in rrlyrae.ids]

tminmax = np.array([[t.min(), t.max()] for t,_,_,_ in data])
print(tminmax.min(0), tminmax.max(0))

counts = np.array([[np.sum(filts == f) for f in 'ugriz']
                   for t, y, dy, filts in data])
counts_partial = np.array([[np.sum(filts == f) for f in 'ugriz']
                           for t, y, dy, filts in data_partial])

print(np.mean(counts, 0), np.median(counts, 0))
print(np.mean(counts_partial, 0), np.median(counts_partial, 0))


fig, ax = plt.subplots(2)
sns.violinplot(counts, ax=ax[0], names='ugriz')
sns.violinplot(counts_partial, ax=ax[1], names='ugriz')

ax[0].set_ylim(20, 100)
ax[1].set_ylim(0, 20)

plt.show()
