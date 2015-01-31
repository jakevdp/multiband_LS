import tarfile

import numpy as np

from gatspy.datasets import RRLyraeGenerated, fetch_rrlyrae

LSST_FILE = 'LSST.specWindow.tar.gz'


class LSSTsims(object):
    def __init__(self, lsst_file=LSST_FILE):
        self.pointings = self._load_sims(lsst_file)
        self.S82data = fetch_rrlyrae()
        
    def _load_sims(self, lsst_file):
        files = tarfile.open(lsst_file, 'r')
        dtype = np.dtype([('ra', 'float32'), ('dec', 'float32'),
                          ('mjd', 'float64'), ('filter', 'int'),
                          ('m5', 'float32')])
        return [np.loadtxt(files.extractfile(member), dtype=dtype)
                for member in files]

    def generate_lc(self, pointing_index, n_days, gmag, S82index,
                    random_state=None):
        gen = RRLyraeGenerated(self.S82data.ids[S82index],
                               random_state=random_state)

        pointing = self.pointings[pointing_index]
        pointing = pointing[pointing['mjd'] <= pointing['mjd'].min() + n_days]
        t = pointing['mjd']
        filts = pointing['filter']
        m5 = pointing['m5']
        
        # generate magnitudes; no errors
        mag = np.zeros_like(t)
        for i, filt in enumerate('ugrizy'):
            mask = (filts == i)
            # HACK: for y-band, use z-band template
            if filt == 'y': filt = 'z'
            mag[mask] = gen.generated(filt, t[mask])

        # adjust mags to desired r-band mean
        gmag_mean = mag[filts == 1].mean()
        mag += (gmag - gmag_mean)

        # compute magnitude error from m5 (eq 5 of Ivezic 2014 LSST paper)
        gamma = np.array([0.037, 0.038, 0.039, 0.039, 0.040, 0.040])
        x = 10 ** (0.4 * (mag - m5))
        sig2_rand = (0.04 - gamma[filts]) * x + gamma[filts] * x ** 2
        sig2_sys = 0.005 ** 2
        dmag = np.sqrt(sig2_sys + sig2_rand)

        rng = np.random.RandomState(random_state)
        mag += dmag * rng.randn(len(mag))

        return t, mag, dmag, filts


if __name__ == '__main__':
    F = LSSTsims()
    t, mag, dmag, filts = F.generate_lc(0, n_days=3650, gmag=24.0, S82index=0)
    print(len(t))

    import matplotlib.pyplot as plt
    import seaborn; seaborn.set()

    # Note: each night has multiple obs.
    for f in np.unique(filts):
        mask = (filts == f)
        plt.errorbar(t[mask], mag[mask], dmag[mask], fmt='.', label='ugrizy'[f])
    plt.legend()
    plt.show()
