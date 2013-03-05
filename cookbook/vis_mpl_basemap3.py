"""
Vis: Plot a map using the Robinson map projection and contours
"""
from fatiando import logger, gridder, utils, vis

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Generate some data to plot
area = (-180, 180, -80, 80)
shape = (100, 100)
lon, lat = gridder.regular(area, shape)
data = utils.gaussian2d(lon, lat, 30, 60, 10, 30, angle=-60)

# Now get a basemap to plot with some projection
bm = vis.mpl.basemap(area, 'robin')

# And now plot everything passing the basemap to the plotting functions
vis.mpl.figure()
vis.mpl.contour(lon, lat, data, shape, 15, basemap=bm)
bm.drawcoastlines()
bm.drawmapboundary(fill_color='aqua')
bm.drawcountries()
bm.fillcontinents(color='coral')
vis.mpl.draw_geolines((-180, 180, -90, 90), 60, 30, bm)
vis.mpl.show()

