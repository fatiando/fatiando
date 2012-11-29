"""
GravMag: Compare the results of different 3D potential field imaging methods
(migration, generalized inverse, and sandwich model)
"""
from multiprocessing import Pool
from fatiando import logger, gridder, mesher, gravmag
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Make some synthetic gravity data from a polygonal prism model
log.info("Draw the polygons one by one")
bounds = [-10000, 10000, -10000, 10000, 0, 10000]
area = bounds[:4]
depths = [0, 1000, 3000, 7000]
prisms = []
for i in range(1, len(depths)):
    # Plot previous prisms
    axes = mpl.figure().gca()
    mpl.axis('scaled')
    for p in prisms:
        mpl.polygon(p, '.-k', xy2ne=True)
    # Draw a new polygon
    polygon = mpl.draw_polygon(area, axes, xy2ne=True)
    # append the newly drawn one
    prisms.append(
        mesher.PolygonalPrism(polygon, depths[i - 1], depths[i],
            {'density':500}))
meshshape = (30, 30, 30)
xp, yp, zp = gridder.regular(area, meshshape[1:], z=-10)
gz = gravmag.polyprism.gz(xp, yp, zp, prisms)

# Plot the data
mpl.figure()
mpl.axis('scaled')
mpl.contourf(yp, xp, gz, meshshape[1:], 30)
mpl.colorbar()
mpl.xlabel('East (km)')
mpl.ylabel('North (km)')
mpl.m2km()
mpl.show()

# A function to the imaging methods and make the 3D plots
def run(title):
    if title == 'Migration':
        result = gravmag.imaging.migrate(xp, yp, zp, gz, bounds[-2], bounds[-1],
            meshshape, power=0.5)
    elif title == 'Generalized Inverse':
        result = gravmag.imaging.geninv(xp, yp, zp, gz, meshshape[1:],
            bounds[-2], bounds[-1], meshshape[0])
    elif title == 'Sandwich':
        result = gravmag.imaging.sandwich(xp, yp, zp, gz, meshshape[1:],
            bounds[-2], bounds[-1], meshshape[0], power=0.5)
    # Plot the results
    myv.figure()
    myv.polyprisms(prisms, 'density', style='wireframe', linewidth=2)
    myv.prisms(result, 'density', edges=False)
    axes = myv.axes(myv.outline(), ranges=[b*0.001 for b in bounds],
        fmt='%.0f')
    myv.wall_bottom(axes.axes.bounds)
    myv.wall_north(axes.axes.bounds)
    myv.title(title)
    myv.show()

titles = ['Migration', 'Generalized Inverse', 'Sandwich']
# Use a pool of workers to run each method in a different process
pool = Pool(3)
# Use map to apply the run function to each title
pool.map(run, titles)

