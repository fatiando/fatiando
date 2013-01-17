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
potential = gravmag.tesseroid.potential(model, lons, lats, heights)

mpl.figure()
bm = mpl.basemap(area, 'ortho')
bm.bluemarble()
mpl.contourf(lons, lats, potential, shape, 15, basemap=bm)
mpl.colorbar()
mpl.show()
