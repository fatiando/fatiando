"""
GravMag: Forward modeling of the gravitational potential and its derivatives
using tesseroids
"""
import time
from fatiando import gravmag, gridder, logger, utils
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

model = [Tesseroid(-60, -55, -30, -27, 0, -500000, props={'density':200}),
         Tesseroid(-66, -62, -18, -12, 0, -300000, props={'density':-500})]
# Show the model before calculating
scene = myv.figure(zdown=False)
myv.tesseroids(model, 'density')
myv.continents(linewidth=2)
myv.earth(opacity=0.8)
myv.meridians(range(0, 360, 45), opacity=0.2)
myv.parallels(range(-90, 90, 45), opacity=0.2)
scene.scene.camera.position = [23175275.131412581, -16937347.013663091, 
    -4924328.2822419703]
scene.scene.camera.focal_point = [0.0, 0.0, 0.0]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.083030001958377356, -0.17178720527713925, 
    0.98162883763562181]
scene.scene.camera.clipping_range = [9229054.5133903362, 54238225.321054712]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
myv.show()

# Create the computation grid
area = (-80, -30, -40, 10)
shape = (50, 50)
lons, lats, heights = gridder.regular(area, shape, z=250000)

log.info('Calculating...')
start = time.time()
fields = [
    gravmag.tesseroid.potential(model, lons, lats, heights),
    gravmag.tesseroid.gx(model, lons, lats, heights),
    gravmag.tesseroid.gy(model, lons, lats, heights),
    gravmag.tesseroid.gz(model, lons, lats, heights),
    gravmag.tesseroid.gxx(model, lons, lats, heights),
    gravmag.tesseroid.gxy(model, lons, lats, heights),
    gravmag.tesseroid.gxz(model, lons, lats, heights),
    gravmag.tesseroid.gyy(model, lons, lats, heights),
    gravmag.tesseroid.gyz(model, lons, lats, heights),
    gravmag.tesseroid.gzz(model, lons, lats, heights)]
print "Time it took: %s" % (utils.sec2hms(time.time() - start))

log.info('Plotting...')
titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
bm = mpl.basemap(area, 'ortho')
for i, field in enumerate(fields):
    mpl.subplot(4, 3, i + 3)
    mpl.title(titles[i])
    mpl.contourf(lons, lats, field, shape, 15, basemap=bm)
    bm.drawcoastlines()
    mpl.colorbar()
mpl.show()

