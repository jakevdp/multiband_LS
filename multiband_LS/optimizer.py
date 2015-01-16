from __future__ import division, print_function

import sys

import numpy as np


class PeriodicOptimizer(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def find_best_period(self):
        raise NotImplementedError()

    def find_top_N_periods(self, N):
        raise NotImplementedError()


class LinearScanOptimizer(PeriodicOptimizer):
    def __init__(self, period_range=(0.2, 1.2),
                 n_zooms=10, verbose=1, step_frac=0.2):
        self.period_range = period_range
        self.n_zooms = n_zooms
        self.verbose = verbose
        self.step_frac = step_frac

    def _compute_candidate_periods(self, model):
        # Compute the candidate periods
        tmin, tmax = np.min(model.t), np.max(model.t)
        omega_min = 2 * np.pi / np.max(self.period_range)
        omega_max = 2 * np.pi / np.min(self.period_range)

        # width in frequency is inverse of the time range
        width = 2 * np.pi / (tmax - tmin)
        omega_step = self.step_frac * width

        omegas = np.arange(omega_min, omega_max + omega_step, omega_step)
        periods = 2 * np.pi / omegas

        return periods

    def find_best_period(self, model):
        periods = self._compute_candidate_periods(model)
        omegas = 2 * np.pi / periods
        omega_step = omegas[1] - omegas[0]

        if self.verbose:
            print("Finding optimal frequency:")
            print(" - Using omega_step = {0:.5f}".format(omega_step))
            print(" - Computing periods at {0:.0f} steps".format(len(periods)))
            sys.stdout.flush()

        score = model.score(periods)
        i = np.argsort(score)[-self.n_zooms:]

        # zoom-in on the peaks and do a second pass
        if self.n_zooms > 0:
            omegas = np.concatenate([np.linspace(omega - 3 * omega_step,
                                                 omega + 3 * omega_step, 500)
                                     for omega in omegas[i]])
            periods = 2 * np.pi / omegas

            if self.verbose:
                print(" - Zooming & computing periods at {0:.0f} further steps"
                      "".format(len(periods)))
                sys.stdout.flush()

            score = model.score(periods)

        return periods[np.argmax(score)]
        
