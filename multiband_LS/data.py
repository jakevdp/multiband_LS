"""
Data downloaders for the multiband_LS stuff
"""
import os
import tarfile
import gzip
from io import BytesIO

import numpy as np

from astroML.datasets.tools import download_with_progress_bar


def set_data_directory(data_directory):
    global DATA_DIRECTORY
    DATA_DIRECTORY = data_directory

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

DATA_DIRECTORY = ''
set_data_directory(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '_data_downloads')))

SESAR_RRLYRAE_URL = 'http://www.astro.washington.edu/users/bsesar/S82_RRLyr/'


class RRLyraeLC(object):
    def __init__(self, filename, dirname='table1'):
        self.data = tarfile.open(filename)
        self.dirname = dirname
        self._metadata = None
        self._obsdata = None

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

    def get_lightcurve(self, star_id, return_1d=False):
        """Get the light curves for the given ID

        Parameters
        ----------
        star_id : int
            A valid integer star id representing an object in the dataset

        Returns
        -------
        t, y, dy : np.ndarrays (if return_1d == False)
            Times, magnitudes, and magnitude errors.
            The shape of each array is [Nobs, 5], where the columns refer
            to [u,g,r,i,z] bands. Non-observations are indicated by NaN.

        t, y, dy, filts : np.ndarrays (if return_1d == True)
            Times, magnitudes, magnitude errors, and filters
            The shape of each array is [Nobs], and non-observations are
            filtered out.
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

        if return_1d:
            t, y, dy, filts = np.broadcast_arrays(t, y, dy,
                                                  ['u', 'g', 'r', 'i', 'z'])
            good = ~np.isnan(t)
            return t[good], y[good], dy[good], filts[good]
        else:
            return t, y, dy

    def get_metadata(self, lcid):
        if self._metadata is None:
            self._metadata = fetch_lc_params()
        i = np.where(self._metadata['id'] == lcid)[0]
        if len(i) == 0:
            raise ValueError("invalid lcid: {0}".format(lcid))
        return self._metadata[i[0]]

    def get_obsmeta(self, lcid):
        if self._obsdata is None:
            self._obsdata = fetch_lc_fit_params()
        i = np.where(self._obsdata['id'] == lcid)[0]
        if len(i) == 0:
            raise ValueError("invalid lcid: {0}".format(lcid))
        return self._obsdata[i[0]]


def fetch_light_curves(data_dir=None):
    """Fetch light curves from Sesar 2010"""
    if data_dir is None:
        data_dir = DATA_DIRECTORY
    save_loc = os.path.join(data_dir, 'table1.tar.gz')
    url = SESAR_RRLYRAE_URL + 'table1.tar.gz'

    if not os.path.exists(save_loc):
        buf = download_with_progress_bar(url)
        open(save_loc, 'bw').write(buf)

    return RRLyraeLC(save_loc)


def fetch_lc_params(data_dir=None):
    """Fetch data from table 2 of Sesar 2010"""
    if data_dir is None:
        data_dir = DATA_DIRECTORY
    save_loc = os.path.join(data_dir, 'table2.dat.gz')
    url = SESAR_RRLYRAE_URL + 'table2.dat.gz'

    if not os.path.exists(save_loc):
        buf = download_with_progress_bar(url)
        open(save_loc, 'wb').write(buf)

    dtype = [('id', 'i'), ('type', 'S2'), ('P', 'f'),
             ('uA', 'f'), ('u0', 'f'), ('uE', 'f'), ('uT', 'f'),
             ('gA', 'f'), ('g0', 'f'), ('gE', 'f'), ('gT', 'f'),
             ('rA', 'f'), ('r0', 'f'), ('rE', 'f'), ('rT', 'f'),
             ('iA', 'f'), ('i0', 'f'), ('iE', 'f'), ('iT', 'f'),
             ('zA', 'f'), ('z0', 'f'), ('zE', 'f'), ('zT', 'f')]

    return np.loadtxt(save_loc, dtype=dtype)


def fetch_lc_fit_params(data_dir=None):
    """Fetch data from table 3 of Sesar 2010"""
    if data_dir is None:
        data_dir = DATA_DIRECTORY
    save_loc = os.path.join(data_dir, 'table3.dat.gz')
    url = SESAR_RRLYRAE_URL + 'table3.dat.gz'

    if not os.path.exists(save_loc):
        buf = download_with_progress_bar(url)
        open(save_loc, 'wb').write(buf)

    dtype = [('id', 'i'), ('RA', 'f'), ('DEC', 'f'), ('rExt', 'f'),
             ('d', 'f'), ('RGC', 'f'),
             ('u', 'f'), ('g', 'f'), ('r', 'f'),
             ('i', 'f'), ('z', 'f'), ('V', 'f'),
             ('ugmin', 'f'), ('ugmin_err', 'f'),
             ('grmin', 'f'), ('grmin_err', 'f')]

    return np.loadtxt(save_loc, dtype=dtype)


class RRLyraeTemplates(object):
    def __init__(self, filename, dirname='table1'):
        self.data = tarfile.open(filename)
        self.dirname = dirname

    @property
    def filenames(self):
        return self.data.getnames()

    @property
    def ids(self):
        return [f.split('.')[0] for f in self.filenames]

    def get_template(self, template_id):
        try:
            data = np.loadtxt(self.data.extractfile(template_id + '.dat'))
        except KeyError:
            raise ValueError("invalid star id: {0}".format(template_id))
        return data.T
    

def fetch_rrlyrae_templates(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIRECTORY
    save_loc = os.path.join(data_dir, 'RRLyr_ugriz_templates.tar.gz')
    url = SESAR_RRLYRAE_URL + 'RRLyr_ugriz_templates.tar.gz'

    if not os.path.exists(save_loc):
        buf = download_with_progress_bar(url)
        open(save_loc, 'wb').write(buf)

    return RRLyraeTemplates(save_loc)
    
