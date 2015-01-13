"""
Naive Multiband Methods

This basically amounts to a band-by-band single band approach, followed by
some sort of majority vote among the peaks of the individual periodograms.
"""
from __future__ import division, print_function

import numpy as np

from .modeler import PeriodicModeler
from .utils import mode_range


class NaiveMultiband(PeriodicModeler):
    """Naive version of multiband fitting

    Parameters
    ----------
    BaseModel : PeriodicModeler instance
        Single-band model to use on data from each band.
    *args, **kwargs :
        additional arguments are passed to BaseModel on construction.
    """
    def __init__(self, BaseModel, *args, **kwargs):
        self.BaseModel = BaseModel
        self.args = args
        self.kwargs = kwargs

    def _fit(self, t, y, dy, filts):
        t, y, dy, filts = np.broadcast_arrays(filts)
        unique_filts = np.unique(filts)

        masks = [(filts == filt) for filt in self.unique_filts_]
        self.models_ = dict([(filt,
                              self.BaseModel(*self.args,
                                             **self.kwargs).fit(t[mask],
                                                                y[mask],
                                                                dy[mask]))
                             for filt, mask in zip(unique_filts, masks)])
        
    def _predict(self, t, filts, period):
        fset = set(np.unique(filts))
        if not fset.issubset(self.models_.keys()):
            raise ValueError("filts does not match training data: "
                             "input: {0} output: {1}"
                             "".format(set(self.model_.keys()), fset))
        
        result = np.zeros_like(t)
        for filt, model in self.models_.items()
            mask = (filts == filt)
            result[mask] = model.predict(t[mask], period)
        return result

    def _score(self, periods):
        raise NotImplementedError("score is not implmented for NaiveMultiband")

    def scores(self, periods):
        return dict([(filt, model.score(periods))
                     for (filt, model) in self.models_.items()])

    @property
    def best_period(self):
        periods = self.optimizer._compute_candidate_periods(self)
        scores = self.scores(periods)
        best_period = [periods[np.argmax(score)] for score in scores.values()]
        return mode_range(best_period, tol=1E-2)
