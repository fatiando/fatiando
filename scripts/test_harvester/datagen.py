"""
Generate the test data.
"""
import numpy
import fatiando as ft
from fatiando.mesher.ddd import Prism

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

bounds = [0, 5000, 0, 5000, -500, 1000]
model = [Prism(600, 1200, 200, 4200, 400, 900, {'density':1000}),
         Prism(3000, 4000, 1000, 2000, 200, 800, {'density':500}),
         Prism(2700, 3200, 3700, 4200, 0, 900, {'density':1500})]
# show it
ft.vis.figure3d()
ft.vis.prisms(model, 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds],
              fmt='%.1f', nlabels=6)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
ft.vis.show3d()
# Generate the data grid
shape = (25, 25)
area = bounds[0:4]
x, y = ft.grd.regular(area, shape)
# Generate synthetic topography
height = ft.utils.contaminate(
          300*ft.utils.gaussian2d(x, y, 1000, 3000, x0=500, y0=1000, angle=-60)
          + 1000*ft.utils.gaussian2d(x, y, 500, 2000, x0=3000, y0=3000),
        0.05, percent=True)
# Calculate the data
noise = 1
noisegz = 0.1
z = -height - 150
data = [x, y, z, height,
    ft.utils.contaminate(ft.pot.prism.gz(x, y, z, model), noisegz),
    ft.utils.contaminate(ft.pot.prism.gxx(x, y, z, model), noise),
    ft.utils.contaminate(ft.pot.prism.gxy(x, y, z, model), noise),
    ft.utils.contaminate(ft.pot.prism.gxz(x, y, z, model), noise),
    ft.utils.contaminate(ft.pot.prism.gyy(x, y, z, model), noise),
    ft.utils.contaminate(ft.pot.prism.gyz(x, y, z, model), noise),
    ft.utils.contaminate(ft.pot.prism.gzz(x, y, z, model), noise)]
with open('data.txt', 'w') as f:
    f.write(ft.log.header(comment='#'))
    f.write("# Noise corrupted gz and tensor components:\n")
    f.write("#   noise = %g Eotvos\n" % (noise))
    f.write("#   noise = %g mGal\n" % (noisegz))
    f.write("#   coordinates are in meters\n")
    f.write("#   gz in mGal and tensor in Eotvos\n")
    f.write("# x   y   z   height   gz   gxx   gxy   gxz   gyy   gyz   gzz\n")
    numpy.savetxt(f, numpy.array(data).T)
# Show it
ft.vis.figure(figsize=(10, 9))
names = "z   height   gz   gxx   gxy   gxz   gyy   gyz   gzz".split()
for i, comp in enumerate(data[2:]):
    ft.vis.subplot(3, 3, i + 1)
    ft.vis.axis('scaled')
    ft.vis.title(names[i])
    levels = ft.vis.contourf(y*0.001, x*0.001, comp, shape, 8)
    ft.vis.contour(y*0.001, x*0.001, comp, shape, levels)
    if i == 3:
        ft.vis.ylabel('North = x (km)')
    if i == 7:        
        ft.vis.xlabel('East = y (km)')
ft.vis.show()
