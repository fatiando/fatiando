"""
Datasets: Fetch the CRUST2.0 model, convert it to tesseroids and calculate its
gravity signal in parallel
"""
import time
from multiprocessing import Pool
from fatiando import gridder, utils, datasets
from fatiando.gravmag import tesseroid
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl, myv

# Get the data from their website and convert it to tesseroids
# Will download the archive and save it with the default name
archive = datasets.fetch_crust2()
model = datasets.crust2_to_tesseroids(archive)

# Plot the tesseroid model
myv.figure(zdown=False)
myv.tesseroids(model, 'density')
myv.continents(linewidth=3)
myv.show()

# Make the computation grid
area = (-180, 180, -80, 80)
shape = (100, 100)
lons, lats, heights = gridder.regular(area, shape, z=250000)

# Divide the model into nproc slices and calculate them in parallel
def calculate(chunk):
    return tesseroid.gz(lons, lats, heights, chunk)
def split(model, nproc):
    chunksize = len(model)/nproc
    for i in xrange(nproc - 1):
        yield model[i*chunksize : (i + 1)*chunksize]
    yield model[(nproc - 1)*chunksize : ]
start = time.time()
nproc = 8
pool = Pool(processes=nproc)
gz = sum(pool.map(calculate, split(model, nproc)))
pool.close()
print "Time it took: %s" % (utils.sec2hms(time.time() - start))

mpl.figure(figsize=(10, 4))
mpl.title('Crust gravity signal at 250km height')
bm = mpl.basemap(area, 'robin')
mpl.contourf(lons, lats, gz, shape, 35, basemap=bm)
cb = mpl.colorbar()
cb.set_label('mGal')
bm.drawcoastlines()
bm.drawmapboundary()
bm.drawparallels(range(-90, 90, 45), labels=[0, 1, 0, 0])
bm.drawmeridians(range(-180, 180, 60), labels=[0, 0, 0, 1])
mpl.show()
