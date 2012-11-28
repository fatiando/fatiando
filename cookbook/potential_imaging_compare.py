"""
Potential: Compare the results of different 3D potential field imaging methods
(migration, generalized inverse, and sandwich model)
"""
from multiprocessing import Pool
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Make some synthetic gravity data from a polygonal prism model
log.info("Draw the polygons one by one")
bounds = [-10000, 10000, -10000, 10000, 0, 10000]
area = bounds[:4]
depths = [0, 1000, 3000, 7000]
prisms = []
for i in range(1, len(depths)):
    # Plot previous prisms
    axes = ft.vis.figure().gca()
    ft.vis.axis('scaled')
    for p in prisms:
        ft.vis.polygon(p, '.-k', xy2ne=True)
    # Draw a new polygon
    polygon = ft.ui.picker.draw_polygon(area, axes, xy2ne=True)
    # append the newly drawn one
    prisms.append(
        ft.msh.ddd.PolygonalPrism(polygon, depths[i - 1], depths[i],
            {'density':500}))
meshshape = (30, 30, 30)
xp, yp, zp = ft.gridder.regular(area, meshshape[1:], z=-10)
gz = ft.pot.polyprism.gz(xp, yp, zp, prisms)

# Plot the data
ft.vis.figure()
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, gz, meshshape[1:], 30)
ft.vis.colorbar()
ft.vis.xlabel('East (km)')
ft.vis.ylabel('North (km)')
ft.vis.m2km()
ft.vis.show()

# A function to the imaging methods and make the 3D plots
def run(title):
    if title == 'Migration':
        result = ft.pot.imaging.migrate(xp, yp, zp, gz, bounds[-2], bounds[-1],
            meshshape, power=0.5)
    elif title == 'Generalized Inverse':
        result = ft.pot.imaging.geninv(xp, yp, zp, gz, meshshape[1:],
            bounds[-2], bounds[-1], meshshape[0])
    elif title == 'Sandwich':
        result = ft.pot.imaging.sandwich(xp, yp, zp, gz, meshshape[1:],
            bounds[-2], bounds[-1], meshshape[0], power=0.5)
    # Plot the results
    ft.vis.figure3d()
    ft.vis.polyprisms(prisms, 'density', style='wireframe', linewidth=2)
    ft.vis.prisms(result, 'density', edges=False)
    axes = ft.vis.axes3d(ft.vis.outline3d(), ranges=[b*0.001 for b in bounds],
                         fmt='%.0f')
    ft.vis.wall_bottom(axes.axes.bounds)
    ft.vis.wall_north(axes.axes.bounds)
    ft.vis.title3d(title)
    ft.vis.show3d()

titles = ['Migration', 'Generalized Inverse', 'Sandwich']
# Use a pool of workers to run each method in a different process
pool = Pool(3)
# Use map to apply the run function to each title
pool.map(run, titles)

