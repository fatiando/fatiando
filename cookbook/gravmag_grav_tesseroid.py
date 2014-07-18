"""
GravMag: Forward modeling of the gravitational potential and its derivatives
using tesseroids
"""
import time
from fatiando import gridder, utils
from fatiando.gravmag import tesseroid
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl, myv

model = [Tesseroid(-60, -55, -30, -27, 0, -500000, props={'density': 200}),
         Tesseroid(-66, -62, -18, -12, 0, -300000, props={'density': -500})]
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
shape = (100, 100)
lons, lats, heights = gridder.regular(area, shape, z=250000)

start = time.time()
fields = [
    tesseroid.potential(lons, lats, heights, model),
    tesseroid.gx(lons, lats, heights, model),
    tesseroid.gy(lons, lats, heights, model),
    tesseroid.gz(lons, lats, heights, model),
    tesseroid.gxx(lons, lats, heights, model),
    tesseroid.gxy(lons, lats, heights, model),
    tesseroid.gxz(lons, lats, heights, model),
    tesseroid.gyy(lons, lats, heights, model),
    tesseroid.gyz(lons, lats, heights, model),
    tesseroid.gzz(lons, lats, heights, model)]
print "Time it took: %s" % (utils.sec2hms(time.time() - start))

titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
bm = mpl.basemap(area, 'merc')
mpl.figure()
mpl.title(titles[0])
mpl.contourf(lons, lats, fields[0], shape, 40, basemap=bm)
bm.drawcoastlines()
mpl.colorbar()
mpl.figure()
for i, field in enumerate(fields[1:4]):
    mpl.subplot(1, 3, i + 1)
    mpl.title(titles[i + 1])
    mpl.contourf(lons, lats, field, shape, 40, basemap=bm)
    bm.drawcoastlines()
    mpl.colorbar()
mpl.figure()
for i, field in enumerate(fields[4:]):
    mpl.subplot(2, 3, i + 1)
    mpl.title(titles[i + 4])
    mpl.contourf(lons, lats, field, shape, 40, basemap=bm)
    bm.drawcoastlines()
    mpl.colorbar()
mpl.show()
