"""
GravMag: Forward modeling of the gravitational potential and its derivatives
using 3D prisms
"""
from fatiando import gravmag, gridder
from fatiando.mesher import Tesseroid
from fatiando.vis import mpl

model = [Tesseroid(-5, 5, -10, 10, 0, -50000, props={'density':200}),
         Tesseroid(-12, -16, -12, -16, 0, -30000, props={'density':-500})]
area = (-30, 30, -30, 30)
shape = (50, 50)
lons, lats, heights = gridder.regular(area, shape, z=250000)
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
titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure()
bm = mpl.basemap(area, 'ortho')
for i, field in enumerate(fields):
    mpl.subplot(4, 3, i + 3)
    bm.bluemarble()
    mpl.contourf(lons, lats, field, shape, 15, basemap=bm)
    mpl.colorbar()
mpl.show()
