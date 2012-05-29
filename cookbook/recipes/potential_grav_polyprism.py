"""
Create synthetic data from a 3D prism with polygonal horizontal crossection.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

area = (-5000, 5000, -5000, 5000)

axes = ft.vis.figure().gca()
prisms = [ft.msh.ddd.PolygonalPrism(ft.ui.picker.draw_polygon(area, axes),
                                    0, 1000, {'density':500})]
shape = (100,100)
xp, yp, zp = ft.grd.regular(area, shape, z=-1)
gz = ft.pot.polyprism.gz(xp, yp, zp, prisms)

ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.title("gz produced by prism model (mGal)")
ft.vis.contourf(xp, yp, gz, shape, 20)
ft.vis.colorbar()
ft.vis.polygon(prisms[0], '.-k', label='top=0km bottom=1km')
ft.vis.legend()
ft.vis.show()
