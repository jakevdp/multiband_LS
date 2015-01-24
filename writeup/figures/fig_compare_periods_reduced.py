"""
Plot figures comparing periods between multiband lomb scargle,
supersmoother, and Sesar 2010 for partial dataset.
"""
import matplotlib.pyplot as plt
from fig_compare_periods import plot_periods, plot_example_lightcurve
from multiband_LS.data import fetch_light_curves

rrlyrae = fetch_light_curves(partial=True)
lcid = list(rrlyrae.ids)[4]

fig, ax = plot_example_lightcurve(rrlyrae, lcid)
fig.savefig('fig08a.pdf')

# XXX: use new results here when available
fig, ax = plot_periods(ssm_file='results_old/partial_supersmoother_g.npy',
                       mbls_file='results/partial_multiband_1_0.npy',
                       rrlyrae=rrlyrae)
fig.savefig('fig08b.pdf')
plt.show()
