"""
Potential: Generate synthetic gravity data on an irregular grid
"""
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

prisms = [ft.msh.ddd.Prism(-2000, 2000, -2000, 2000, 0, 2000, {'density':1000})]
xp, yp, zp = ft.gridder.scatter((-5000, 5000, -5000, 5000), n=100, z=-100)
gz = ft.pot.prism.gz(xp, yp, zp, prisms)

shape = (100,100)
ft.vis.axis('scaled')
ft.vis.title("gz produced by prism model on an irregular grid (mGal)")
ft.vis.plot(xp, yp, '.k', label='Grid points')
levels = ft.vis.contourf(xp, yp, gz, shape, 12, interp=True)
ft.vis.contour(xp, yp, gz, shape, levels, interp=True)
ft.vis.legend(loc='lower right', numpoints=1)
ft.vis.m2km()
ft.vis.show()
