"""
Vis: Plot a map using the Robinson map projection and contours
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Generate some data to plot
area = (-180, 180, -80, 80)
shape = (100, 100)
lon, lat = ft.gridder.regular(area, shape)
data = ft.utils.gaussian2d(lon, lat, 30, 60, 10, 30, angle=-60)

# Now get a basemap to plot with some projection
bm = ft.vis.basemap(area, 'robin')

# And now plot everything passing the basemap to the plotting functions
ft.vis.figure()
ft.vis.contour(lon, lat, data, shape, 15, basemap=bm)
bm.drawcoastlines()
bm.drawmapboundary(fill_color='aqua')
bm.drawcountries()
bm.fillcontinents(color='coral')
ft.vis.draw_geolines((-180, 180, -90, 90), 60, 30, bm)
ft.vis.show()

