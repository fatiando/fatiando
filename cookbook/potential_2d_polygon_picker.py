"""
Potential: 2D forward modeling with polygons
"""
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Notice that the last two number are switched.
# This way, the z axis in the plots points down.
area = (-5000, 5000, 5000, 0)
axes = ft.vis.figure().gca()
ft.vis.xlabel("X")
ft.vis.ylabel("Z")
ft.vis.axis('scaled')
polygons = [ft.msh.dd.Polygon(ft.ui.picker.draw_polygon(area, axes),
                              {'density':500})]
xp = numpy.arange(-4500, 4500, 100)
zp = numpy.zeros_like(xp)
gz = ft.pot.talwani.gz(xp, zp, polygons)

ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.subplot(2,1,1)
ft.vis.title(r"Gravity anomaly produced by the model")
ft.vis.plot(xp, gz, '-k', linewidth=2)
ft.vis.ylabel("mGal")
ft.vis.xlim(-5000, 5000)
ft.vis.subplot(2,1,2)
ft.vis.polygon(polygons[0], 'o-k', linewidth=2, fill='k', alpha=0.5)
ft.vis.xlabel("X")
ft.vis.ylabel("Z")
ft.vis.set_area(area)
ft.vis.show()
