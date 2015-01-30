import numpy as np
import matplotlib.pyplot as plt

from compute_results import gather_results
from gatspy.datasets import fetch_rrlyrae

rng = np.random.RandomState(0)
template_indices = rng.randint(0, 483, 5)
pointing_indices = np.arange(1, 24)[:, None]
ndays = np.array([365, 3 * 365, 5 * 365, 10 * 365])[:, None, None]
rmags = np.array([20, 22, 24.5])[:, None, None, None]

results = gather_results('results.npy',
                         pointing_indices=pointing_indices,
                         ndays=ndays,
                         rmags=rmags,
                         template_indices=template_indices)

rrlyrae = fetch_rrlyrae()
periods = np.asarray([rrlyrae.get_metadata(rrlyrae.ids[0])['P']
                      for i in template_indices.ravel()])

print(results.shape)
match = np.any((abs(results - periods[:, None]) < 0.01), -1)
print(match.shape)

frac = match.sum((-1, -2)) / np.prod(match.shape[-2:])
print(frac.shape)

rmags = rmags.ravel()
ndays = ndays.ravel()
for i in range(len(rmags)):
    plt.plot(ndays, frac[i], label='r = {0:.2f}'.format(rmags[i]))

plt.legend()
plt.show()
