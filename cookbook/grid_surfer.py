"""
Gridding: Load a Surfer ASCII grid file
"""
from fatiando import datasets, gridder
from fatiando.vis import mpl

# Fetching Bouguer anomaly model data (Surfer ASCII grid file)"
# Will download the archive and save it with the default name
archive = datasets.fetch_bouguer_alps_egm()

# Load the GRD file and convert in three numpy-arrays (x, y, bouguer)
x, y, bouguer, shape = gridder.load_surfer(archive, fmt='ascii')

mpl.figure()
mpl.axis('scaled')
mpl.title("Data loaded from a Surfer ASCII grid file")
mpl.contourf(y, x, bouguer, shape, 15, cmap=mpl.cm.RdBu_r)
cb = mpl.colorbar()
cb.set_label('mGal')
mpl.xlabel('y points to East (km)')
mpl.ylabel('x points to North (km)')
mpl.m2km()
mpl.show()
