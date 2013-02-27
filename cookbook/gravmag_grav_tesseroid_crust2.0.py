"""
GravMag: Calculate the gravity anomaly of the CRUST2.0 model using tesseroids
"""
import time
from multiprocessing import Pool
from fatiando import gravmag, gridder, logger, utils, io
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# Get the data from their website and convert it to tesseroids
# Will download the archive and save it with the default name
log.info("Fetching CRUST2.0 model")
archive = io.fetch_crust2()
log.info("Converting to tesseroids")
model = io.crust2_to_tesseroids(archive)

# Plot the tesseroid model
myv.figure(zdown=False)
myv.tesseroids(model, 'density')
myv.continents(linewidth=3)
myv.show()

# Make the computation grid
area = (0, 360, -80, 80)
shape = (200, 200)
lons, lats, heights = gridder.regular(area, shape, z=250000)

# Divide the model into nproc slices and calculate them in parallel
log.info('Calculating...')
def calculate(chunk):
    return gravmag.tesseroid.gz(chunk, lons, lats, heights)
start = time.time()
nproc = 8 # Model size must be divisible by nproc
chunksize = len(model)/nproc
pool = Pool(processes=nproc)
gz = sum(pool.map(calculate, 
    [model[i*chunksize:(i + 1)*chunksize] for i in xrange(nproc)]))
pool.close()
print "Time it took: %s" % (utils.sec2hms(time.time() - start))

log.info('Plotting...')
mpl.figure()
bm = mpl.basemap(area, 'robin')
bm.bluemarble()
mpl.contourf(lons, lats, gz, shape, 35, basemap=bm)
mpl.colorbar()
mpl.show()
