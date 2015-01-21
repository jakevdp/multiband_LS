"""General Periodic Modeling for Astronomical Time Series"""

__version__ = "0.1"

from .lomb_scargle import LombScargle, LombScargleAstroML
from .lomb_scargle_multiband import (LombScargleMultiband,
                                     LombScargleMultibandFast)
from .supersmoother import SuperSmoother, SuperSmootherMultiband
from .naive_multiband import NaiveMultiband
