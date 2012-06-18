"""
Create synthetic gravity data from a stack of 3D polygonal prisms.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Draw the polygons one by one")
bounds = [-10000, 10000, -10000, 10000, 0, 5000]
area = bounds[:4]
depths = [0, 1000, 2000, 3000, 4000]
prisms = []
for i in range(1, len(depths)):
    axes = ft.vis.figure().gca()
    ft.vis.axis('scaled')
    for p in prisms:
        ft.vis.polygon(p, '.-k', xy2ne=True)
    prisms.append(
        ft.msh.ddd.PolygonalPrism(
            ft.ui.picker.draw_polygon(area, axes, xy2ne=True),
            depths[i - 1], depths[i], {'density':500}))
# Calculate the effect
shape = (100, 100)
xp, yp, zp = ft.grd.regular(area, shape, z=-1)
gz = ft.pot.polyprism.gz(xp, yp, zp, prisms)
# and plot it
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.title("gz produced by prism model (mGal)")
ft.vis.contourf(yp, xp, gz, shape, 20)
ft.vis.colorbar()
for p in prisms:
    ft.vis.polygon(p, '.-k', xy2ne=True)
ft.vis.set_area(area)
ft.vis.show()
# Show the prisms
ft.vis.figure3d()
ft.vis.polyprisms(prisms, 'density')
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[i*0.001 for i in bounds])
ft.vis.show3d()
