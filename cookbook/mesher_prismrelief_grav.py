"""
Meshing: Generate a topographic model using prisms and calculate its gravity
anomaly
"""
from fatiando import gridder, utils, mesher
from fatiando.gravmag import prism
from fatiando.vis import myv, mpl

area = (-150, 150, -300, 300)
shape = (30, 15)
x, y = gridder.regular(area, shape)
height = (-80*utils.gaussian2d(x, y, 100, 200, x0=-50, y0=-100, angle=-60) +
          200*utils.gaussian2d(x, y, 50, 100, x0=80, y0=170))

nodes = (x, y, -1*height)
relief = mesher.PrismRelief(0, gridder.spacing(area,shape), nodes)
relief.addprop('density', (2670 for i in xrange(relief.size)))

gridarea = (-80, 80, -220, 220)
gridshape = (100, 100)
xp, yp, zp = gridder.regular(gridarea, gridshape, z=-200)
gz = prism.gz(xp, yp, zp, relief)

mpl.figure(figsize=(10,7))
mpl.subplot(1, 2, 1)
mpl.title("Synthetic topography")
mpl.axis('scaled')
mpl.pcolor(x, y, height, shape)
cb = mpl.colorbar()
cb.set_label("meters")
mpl.square(gridarea, label='Computation grid')
mpl.legend()
mpl.subplot(1, 2, 2)
mpl.title("Topographic effect")
mpl.axis('scaled')
mpl.pcolor(xp, yp, gz, gridshape)
cb = mpl.colorbar()
cb.set_label("mGal")
mpl.show()

myv.figure()
myv.prisms(relief, prop='density')
axes = myv.axes(myv.outline())
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
