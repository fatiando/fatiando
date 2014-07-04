"""
Gridding: Load a Surfer ASCII grid file
"""
from fatiando import datasets, gridder, gravmag
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

## Now do the upward continuation using the analytical formula
height = 250000
area = [min(x), max(x), min(y), max(y)]
dims = gridder.spacing(area, shape)
bouguercont = gravmag.transform.upcontinue(bouguer, height, x, y, dims)

# Plotthe upward continuation
mpl.figure()
mpl.axis('scaled')
mpl.title("Continued to %d km height" % ((height/1000)))
mpl.contourf(y, x, bouguercont, shape, 15)
cb = mpl.colorbar()
cb.set_label('mGal')
mpl.xlabel('y points to East (km)')
mpl.ylabel('x points to North (km)')
mpl.m2km()
mpl.show()

# Name of the output file
fname = ('bouguer_alps_egm08_height_%d_km.grd' % ((height/1000)))
# Saving the upward continued data in a Surfer ascii grid file
gridder.dump_surfer(fname,bouguercont, shape, x, y, fmt='ascii')
