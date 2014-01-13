"""
GravMag: Forward modeling of the gravity anomaly using tesseroids in parallel
using ``multiprocessing``
"""
import time
from multiprocessing import Pool
from fatiando import gridder, utils
from fatiando.gravmag import tesseroid
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl, myv

# Make a "crust" model with some thinker crust and variable density
marea = (-70, 70, -70, 70)
mshape = (200, 200)
mlons, mlats = gridder.regular(marea, mshape)
dlon, dlat = gridder.spacing(marea, mshape)
depths = (30000 +
    70000*utils.gaussian2d(mlons, mlats, 10, 10, -20, -20) +
    20000*utils.gaussian2d(mlons, mlats, 5, 5, 20, 20))
densities = (2700 +
    500*utils.gaussian2d(mlons, mlats, 40, 40, -20, -20) +
    -300*utils.gaussian2d(mlons, mlats, 20, 20, 20, 20))
model = [
    Tesseroid(lon - 0.5*dlon, lon + 0.5*dlon, lat - 0.5*dlat, lat + 0.5*dlat,
              0, -depth, props={'density':density})
    for lon, lat, depth, density in zip(mlons, mlats, depths, densities)]

# Plot the tesseroid model
myv.figure(zdown=False)
myv.tesseroids(model, 'density')
myv.continents()
myv.earth(opacity=0.7)
myv.show()

# Make the computation grid
area = (-50, 50, -50, 50)
shape = (100, 100)
lons, lats, heights = gridder.regular(area, shape, z=250000)

# Divide the model into nproc slices and calculate them in parallel
def calculate(chunk):
    return tesseroid.gz(lons, lats, heights, chunk)
start = time.time()
nproc = 8 # Model size must be divisible by nproc
chunksize = len(model)/nproc
pool = Pool(processes=nproc)
gz = sum(pool.map(calculate,
    [model[i*chunksize:(i + 1)*chunksize] for i in xrange(nproc)]))
pool.close()
print "Time it took: %s" % (utils.sec2hms(time.time() - start))

mpl.figure()
bm = mpl.basemap(area, 'ortho')
bm.bluemarble()
mpl.contourf(lons, lats, gz, shape, 35, basemap=bm)
mpl.colorbar()
mpl.show()
