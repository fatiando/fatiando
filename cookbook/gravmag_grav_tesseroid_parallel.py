"""
GravMag: Forward modeling of the gravitational potential and its derivatives
using tesseroids and multiprocessing
"""
import time
from multiprocessing import Pool
from fatiando import gravmag, gridder, logger, utils
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())

model = [Tesseroid(-5, 5, -10, 10, 0, -50000, props={'density':200}),
         Tesseroid(-12, -16, -12, -16, 0, -30000, props={'density':-500})]
area = (-30, 30, -30, 30)
shape = (50, 50)
lons, lats, heights = gridder.regular(area, shape, z=250000)

log.info('Calculating...')
start = time.time()
def calculate(func):
    return func(model, lons, lats, heights)
functions = [
    gravmag.tesseroid.potential,
    gravmag.tesseroid.gx,
    gravmag.tesseroid.gy,
    gravmag.tesseroid.gz,
    gravmag.tesseroid.gxx,
    gravmag.tesseroid.gxy,
    gravmag.tesseroid.gxz,
    gravmag.tesseroid.gyy,
    gravmag.tesseroid.gyz,
    gravmag.tesseroid.gzz]
pool = Pool(processes=10)
fields = pool.map(calculate, functions)
pool.close()
print "Time it took: %s" % (utils.sec2hms(time.time() - start))

log.info('Plotting...')
titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
bm = mpl.basemap(area, 'ortho')
for i, field in enumerate(fields):
    mpl.subplot(4, 3, i + 3)
    mpl.title(titles[i])
    bm.bluemarble()
    mpl.contourf(lons, lats, field, shape, 15, basemap=bm)
    mpl.colorbar()
mpl.show()
