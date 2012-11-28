"""
GravMag: 3D forward modeling of total-field magnetic anomaly using polygonal
prisms
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

log.info("Draw the polygons one by one")
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
area = bounds[:4]
axis = ft.vis.figure().gca()
ft.vis.axis('scaled')
prisms = [
    ft.mesher.PolygonalPrism(
        ft.vis.map.draw_polygon(area, axis, xy2ne=True),
        0, 2000, {'magnetization':2})]
# Calculate the effect
shape = (100, 100)
xp, yp, zp = ft.gridder.regular(area, shape, z=-500)
tf = ft.gravmag.polyprism.tf(xp, yp, zp, prisms, 30, -15)
# and plot it
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.title("Total field anomalyproduced by prism model (nT)")
ft.vis.contourf(yp, xp, tf, shape, 20)
ft.vis.colorbar()
for p in prisms:
    ft.vis.polygon(p, '.-k', xy2ne=True)
ft.vis.set_area(area)
ft.vis.m2km()
ft.vis.show()
# Show the prisms
ft.vis.figure3d()
ft.vis.polyprisms(prisms, 'magnetization')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds])
ft.vis.wall_north(bounds)
ft.vis.wall_bottom(bounds)
ft.vis.show3d()
