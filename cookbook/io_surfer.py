"""
I/O: Load a Surfer ASCII grid file
"""
from fatiando import io, gravmag, gridder
from fatiando.vis import mpl
import numpy as np

# Get the data from their website
# Will download the archive and save it with the default name
print "Fetching Bouguer anomaly model (Surfer ASCII grid file)"
archive = io.fetch_bouguer_alps_egm()

# Load the GRD file and convert in three numpy-arrays (y, x, bouguer)
print "Loading the GRD file..."
y, x, bouguer, shape = io.load_surfer(archive, fmt='ascii')

area = [min(x), max(x), min(y), max(y)]

print "Plotting..."
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
dims = gridder.spacing(area, shape)
bouguercont = gravmag.transform.upcontinue(bouguer, height, x, y, dims)

print "Plotting the upward continuation..."
mpl.figure()
mpl.axis('scaled')
mpl.title("Continued to %d km height" % ((height/1000)))
mpl.contourf(y, x, bouguercont, shape, 15)
cb = mpl.colorbar()
cb.set_label('mGal')
mpl.xlabel('y points to East (km)')
mpl.ylabel('x points to North (km)')
mpl.m2km()
#mpl.show()

# Name of the output file
fname = ('bouguer_alps_egm08_height_%d_km.grd' % ((height/1000)))
io.dump_surfer(fname,bouguercont, shape, x, y)

# Load the GRD file and convert in three numpy-arrays (y, x, bouguer)
print "Loading the GRD file..."
archive = str(('bouguer_alps_egm08_height_%d_km.grd' % (height/1000)))
y, x, bouguercont2, shape = io.load_surfer(archive, fmt='ascii')

print "Plotting..."
mpl.figure()
mpl.axis('scaled')
mpl.title("Data loaded from a Surfer ASCII grid file")
mpl.contourf(y, x, bouguercont2, shape, 15)
cb = mpl.colorbar()
cb.set_label('mGal')
mpl.xlabel('y points to East (km)')
mpl.ylabel('x points to North (km)')
mpl.m2km()
mpl.show()
