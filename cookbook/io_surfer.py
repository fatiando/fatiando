"""
I/O: Load a Surfer ASCII grid file
"""
from fatiando import logger, io
from fatiando.vis import mpl
import urllib

log = logger.get()
log.info(logger.header())

# Get the data from their website
# Will download the archive and save it with the default name
log.info("Fetching Bouguer anomaly model (Surfer ASCII grid file)")
archive = io.fetch_bouguer_alps_egm()
# Load the GRD file and convert in three numpy-arrays (y, x, bouguer)
log.info('Loading the GRD file...')
y, x, bouguer, shape = io.load_surfer(archive, fmt='ascii')

log.info('Plotting...')
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