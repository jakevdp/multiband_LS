from multiband_LS import LombScargleMultiband, SuperSmoother

# Fetch the Sesar 2010 RR Lyrae data
from multiband_LS.data import fetch_light_curves
data = fetch_light_curves()
t, mag, dmag, filts = data.get_lightcurve(data.ids[0],
                                          return_1d=True)

# Construct the multiband model
model = LombScargleMultiband(Nterms_base=0, Nterms_band=1)
model.fit(t, mag, dmag, filts)

# Compute power at the following periods
model.periodogram([0.2, 0.3, 0.4])

# Construct the supersmoother model
model = SuperSmoother()
gband = (filts == 'g')
model.fit(t[gband], mag[gband], dmag[gband])

# Compute power at the following periods
model.periodogram([0.2, 0.3, 0.4])
