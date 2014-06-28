"""
Generate the test data.
"""
import numpy

from fatiando import gravmag, logger, gridder, utils
from fatiando.vis import mpl, myv
from fatiando.mesher import Prism


log = logger.get()
log.info(logger.header())
log.info(__doc__)

bounds = [0, 5000, 0, 5000, -500, 1000]
model = [Prism(600, 1200, 200, 4200, 400, 900, {'density': 1000}),
         Prism(3000, 4000, 1000, 2000, 200, 800, {'density': 500}),
         Prism(2700, 3200, 3700, 4200, 0, 900, {'density': 1500})]
# show it
myv.figure()
myv.prisms(model, 'density')
myv.axes(myv.outline(bounds), ranges=[i * 0.001 for i in bounds], fmt='%.1f',
         nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
# Generate the data grid
shape = (25, 25)
area = bounds[0:4]
x, y = gridder.regular(area, shape)
# Generate synthetic topography
height = (300 * utils.gaussian2d(x, y, 1000, 3000, x0=500, y0=1000, angle=-60)
          + 1000 * utils.gaussian2d(x, y, 500, 2000, x0=3000, y0=3000))
# Calculate the data
noise = 1
noisegz = 0.1
z = -height - 150
data = [x, y, z, height,
        utils.contaminate(gravmag.prism.gz(x, y, z, model), noisegz),
        utils.contaminate(gravmag.prism.gxx(x, y, z, model), noise),
        utils.contaminate(gravmag.prism.gxy(x, y, z, model), noise),
        utils.contaminate(gravmag.prism.gxz(x, y, z, model), noise),
        utils.contaminate(gravmag.prism.gyy(x, y, z, model), noise),
        utils.contaminate(gravmag.prism.gyz(x, y, z, model), noise),
        utils.contaminate(gravmag.prism.gzz(x, y, z, model), noise)]
with open('data.txt', 'w') as f:
    f.write(logger.header(comment='#'))
    f.write("# Noise corrupted gz and tensor components:\n")
    f.write("#   noise = %g Eotvos\n" % (noise))
    f.write("#   noise = %g mGal\n" % (noisegz))
    f.write("#   coordinates are in meters\n")
    f.write("#   gz in mGal and tensor in Eotvos\n")
    f.write("# x   y   z   height   gz   gxx   gxy   gxz   gyy   gyz   gzz\n")
    numpy.savetxt(f, numpy.transpose(data))
# Show it
mpl.figure(figsize=(10, 9))
names = "z   height   gz   gxx   gxy   gxz   gyy   gyz   gzz".split()
for i, comp in enumerate(data[2:]):
    mpl.subplot(3, 3, i + 1)
    mpl.axis('scaled')
    mpl.title(names[i])
    levels = mpl.contourf(y * 0.001, x * 0.001, comp, shape, 8)
    mpl.contour(y * 0.001, x * 0.001, comp, shape, levels)
    if i == 3:
        mpl.ylabel('North = x (km)')
    if i == 7:
        mpl.xlabel('East = y (km)')
mpl.show()
