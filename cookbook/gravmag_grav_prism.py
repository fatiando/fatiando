"""
GravMag: Forward modeling of the gravitational potential and its derivatives
using 3D prisms
"""
from fatiando import logger, mesher, gridder, gravmag
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())
log.info(__doc__)

prisms = [mesher.Prism(-4000,-3000,-4000,-3000,0,2000,{'density':1000}),
          mesher.Prism(-1000,1000,-1000,1000,0,2000,{'density':-900}),
          mesher.Prism(2000,4000,3000,4000,0,2000,{'density':1300})]
shape = (100,100)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
log.info("Calculating fileds...")
fields = [gravmag.prism.potential(xp, yp, zp, prisms),
          gravmag.prism.gx(xp, yp, zp, prisms),
          gravmag.prism.gy(xp, yp, zp, prisms),
          gravmag.prism.gz(xp, yp, zp, prisms),
          gravmag.prism.gxx(xp, yp, zp, prisms),
          gravmag.prism.gxy(xp, yp, zp, prisms),
          gravmag.prism.gxz(xp, yp, zp, prisms),
          gravmag.prism.gyy(xp, yp, zp, prisms),
          gravmag.prism.gyz(xp, yp, zp, prisms),
          gravmag.prism.gzz(xp, yp, zp, prisms)]
log.info("Plotting...")
titles = ['potential', 'gx', 'gy', 'gz',
          'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
mpl.figure(figsize=(8, 9))
mpl.subplots_adjust(left=0.03, right=0.95, bottom=0.05, top=0.92, hspace=0.3)
mpl.suptitle("Potential fields produced by a 3 prism model")
for i, field in enumerate(fields):
    mpl.subplot(4, 3, i + 3)
    mpl.axis('scaled')
    mpl.title(titles[i])
    levels = mpl.contourf(yp*0.001, xp*0.001, field, shape, 15)
    cb = mpl.colorbar()
    mpl.contour(yp*0.001, xp*0.001, field, shape, levels, clabel=False, linewidth=0.1)
mpl.show()

myv.figure()
myv.prisms(prisms, prop='density')
axes = myv.axes(myv.outline())
myv.wall_bottom(axes.axes.bounds, opacity=0.2)
myv.wall_north(axes.axes.bounds)
myv.show()
