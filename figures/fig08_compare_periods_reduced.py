"""
Plot figures comparing periods between multiband lomb scargle,
supersmoother, and Sesar 2010 for reduced dataset.
"""
import matplotlib.pyplot as plt
from fig07_compare_periods import plot_periods, plot_example_lightcurve
from gatspy.datasets import fetch_rrlyrae

rrlyrae = fetch_rrlyrae(partial=True)
lcid = list(rrlyrae.ids)[482]

fig, ax = plot_example_lightcurve(rrlyrae, lcid)
fig.savefig('fig08a.pdf')

fig, ax = plot_periods(ssm_file='S82sims/res_partial_supersmoother_g.npy',
                       mbls_file='S82sims/res_partial_multiband_1_0.npy',
                       rrlyrae=rrlyrae)
fig.savefig('fig08b.pdf')
plt.show()
