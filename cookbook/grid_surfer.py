"""
Gridding: Load/write a Surfer ASCII grid file
"""
from fatiando import datasets, gridder, utils
from fatiando.gravmag import transform, fourier
from fatiando.vis import mpl

# Fetching Bouguer anomaly model data (Surfer ASCII grid file)"
# Will download the archive and save it with the default name
archive = datasets.fetch_bouguer_alps_egm()

# Load the GRD file and convert in three numpy-arrays (y, x, bouguer)
y, x, bouguer, shape = gridder.load_surfer(archive, fmt='ascii')

mpl.figure()
mpl.axis('scaled')
mpl.title("Data loaded from a Surfer ASCII grid file")
mpl.contourf(y, x, bouguer, shape, 15)
cb = mpl.colorbar()
cb.set_label('mGal')
mpl.xlabel('y points to East (km)')
mpl.ylabel('x points to North (km)')
mpl.m2km()
mpl.show()

# Calculate the vertical derivative of the gravity anomaly using FFT
# Need to convert gz to SI units so that the result can be converted to Eotvos
gzz = utils.si2eotvos(fourier.derivz(y, x, utils.mgal2si(bouguer), shape))

mpl.figure()
mpl.axis('scaled')
mpl.title("Vertical derivative of Bouguer anomaly")
mpl.contourf(y, x, gzz, shape, 20, vmin=-200, vmax=200)
cb = mpl.colorbar()
cb.set_label(r'$E\"otv\"os$')
mpl.xlabel('y points to East (km)')
mpl.ylabel('x points to North (km)')
mpl.m2km()
mpl.show()

# Name of the output file
fname = ('gzz_bouguer_alps_egm08.grd')
# Saving the upward continued data in a Surfer ascii grid file
gridder.dump_surfer(fname,gzz, shape, x, y, fmt='ascii')
