"""
GravMag: Calculate the gravity gradient tensor invariants
"""
import fatiando as ft
import numpy

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

log.info("Draw the polygons one by one")
area = [-10000, 10000, -10000, 10000]
dataarea = [-5000, 5000, -5000, 5000]
prisms = []
for depth in [5000, 5000, 5000, 2000]:
    fig = ft.vis.figure()
    ft.vis.axis('scaled')
    ft.vis.square(dataarea)
    for p in prisms:
        ft.vis.polygon(p, '.-k', xy2ne=True)
    ft.vis.set_area(area)
    prisms.append(
        ft.mesher.PolygonalPrism(
            ft.vis.map.draw_polygon(area, fig.gca(), xy2ne=True),
            0, depth, {'density':500}))
# Calculate the effect
shape = (100, 100)
xp, yp, zp = ft.gridder.regular(dataarea, shape, z=-500)
tensor = [
    ft.gravmag.polyprism.gxx(xp, yp, zp, prisms),
    ft.gravmag.polyprism.gxy(xp, yp, zp, prisms),
    ft.gravmag.polyprism.gxz(xp, yp, zp, prisms),
    ft.gravmag.polyprism.gyy(xp, yp, zp, prisms),
    ft.gravmag.polyprism.gyz(xp, yp, zp, prisms),
    ft.gravmag.polyprism.gzz(xp, yp, zp, prisms)]
# Calculate the 3 invariants
invariants = ft.gravmag.tensor.invariants(tensor)
data = tensor + invariants
# and plot it
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.suptitle("Tensor and invariants produced by prism model (Eotvos)")
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'I1', 'I2', 'I']
for i in xrange(len(data)):
    ft.vis.subplot(3, 3, i + 1)
    ft.vis.title(titles[i])
    levels = 20
    if i == 8:
        levels = numpy.linspace(0, 1, levels)
    ft.vis.contourf(yp, xp, data[i], shape, levels)
    ft.vis.colorbar()
    for p in prisms:
        ft.vis.polygon(p, '.-k', xy2ne=True)
    ft.vis.set_area(dataarea)
    ft.vis.m2km()
ft.vis.show()
