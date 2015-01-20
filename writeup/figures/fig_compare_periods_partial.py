"""
Plot figures comparing periods between multiband lomb scargle,
supersmoother, and Sesar 2010 for partial dataset.
"""
import matplotlib.pyplot as plt
from fig_compare_periods import plot_periods


if __name__ == '__main__':
    fig, ax = plot_periods(ssm_file='results/partial_supersmoother_g.npy',
                           mbls_file='results/partial_multiband_1_0.npy')
    fig.savefig('fig08.pdf')
    plt.show()
