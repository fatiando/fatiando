"""
Vis: Plot a map using the Mercator map projection and pseudo-color
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Generate some data to plot
area = (-20, 40, 20, 80)
shape = (100, 100)
lon, lat = ft.grd.regular(area, shape)
data = ft.utils.gaussian2d(lon, lat, 10, 20, 10, 60, angle=45)

# Now get a basemap to plot with some projection
bm = ft.vis.basemap(area, 'merc')

# And now plot everything passing the basemap to the plotting functions
ft.vis.figure(figsize=(5,8))
ft.vis.pcolor(lon, lat, data, shape, basemap=bm)
ft.vis.colorbar()
bm.drawcoastlines()
bm.drawmapboundary()
bm.drawcountries()
ft.vis.draw_geolines(area, 10, 10, bm)
ft.vis.show()

