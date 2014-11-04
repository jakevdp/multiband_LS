"""
Data downloaders for the multiband_LS stuff
"""
import os
import tarfile

import numpy as np

from astroML.datasets.tools import download_with_progress_bar


def set_data_directory(data_directory):
    global DATA_DIRECTORY
    DATA_DIRECTORY = data_directory

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

DATA_DIRECTORY = ''
set_data_directory(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'data')))

SESAR_RRLYRAE_URL = 'http://www.astro.washington.edu/users/bsesar/S82_RRLyr/'


class RRLyraeLC(object):
    def __init__(self, filename, dirname='table1'):
        self.data = tarfile.open(filename)
        self.dirname = dirname

    @property
    def filenames(self):
        return self.data.getnames()

    @property
    def ids(self):
        for f in self.filenames:
            if '/' not in f:
                continue
            f = f.split('/')[1].split('.')
            if len(f) == 1:
                continue
            else:
                yield int(f[0])

    def get_lightcurve(self, star_id):
        """Get the light curves for the given ID

        Parameters
        ----------
        star_id : int
            A valid integer star id representing an object in the dataset

        Returns
        -------
        t, y, dy : np.ndarray
            Times, magnitudes, and magnitude errors.
            The shape of each array is [Nobs, 5], where the columns refer
            to [u,g,r,i,z] bands. Non-observations are indicated by NaN.
        """
        filename = '{0}/{1}.dat'.format(self.dirname, star_id)

        try:
            data = np.loadtxt(self.data.extractfile(filename))
        except KeyError:
            raise ValueError("invalid star id: {0}".format(star_id))

        RA = data[:, 0]
        DEC = data[:, 1]

        t = data[:, 2::3]
        y = data[:, 3::3]
        dy = data[:, 4::3]

        nans = (y == -99.99)
        t[nans] = np.nan
        y[nans] = np.nan
        dy[nans] = np.nan

        return t, y, dy


def fetch_light_curves(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIRECTORY
    save_loc = os.path.join(data_dir, 'table1.tar.gz')
    url = SESAR_RRLYRAE_URL + 'table1.tar.gz'

    if not os.path.exists(save_loc):
        buf = download_with_progress_bar(url)
        open(save_loc, 'bw').write(buf)

    return RRLyraeLC(save_loc)
