"""
Potential: Generate synthetic gradient tensor data from polygonal prisms
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

log.info("Draw the polygons one by one")
bounds = [-10000, 10000, -10000, 10000, 0, 5000]
area = bounds[:4]
axis = ft.vis.figure().gca()
ft.vis.axis('scaled')
prisms = [
    ft.mesher.PolygonalPrism(
        ft.vis.map.draw_polygon(area, axis, xy2ne=True),
        0, 1000, {'density':500})]
# Calculate the effect
shape = (100, 100)
xp, yp, zp = ft.gridder.regular(area, shape, z=-500)
tensor = [
    ft.pot.polyprism.gxx(xp, yp, zp, prisms),
    ft.pot.polyprism.gxy(xp, yp, zp, prisms),
    ft.pot.polyprism.gxz(xp, yp, zp, prisms),
    ft.pot.polyprism.gyy(xp, yp, zp, prisms),
    ft.pot.polyprism.gyz(xp, yp, zp, prisms),
    ft.pot.polyprism.gzz(xp, yp, zp, prisms)]
# and plot it
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.suptitle("Gravity tensor produced by prism model (Eotvos)")
for i in xrange(len(tensor)):
    ft.vis.subplot(3, 2, i + 1)
    ft.vis.title(titles[i])
    ft.vis.contourf(yp, xp, tensor[i], shape, 20)
    ft.vis.colorbar()
    for p in prisms:
        ft.vis.polygon(p, '.-k', xy2ne=True)
    ft.vis.set_area(area)
    ft.vis.m2km()
ft.vis.show()
# Show the prisms
ft.vis.figure3d()
ft.vis.polyprisms(prisms, 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds])
ft.vis.wall_north(bounds)
ft.vis.wall_bottom(bounds)
ft.vis.show3d()
