"""
Plot a map using some map projection and the matplotlib basemap toolkit
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
area = (-20, 80, 20, 80)
bm = ft.vis.basemap(area, 'ortho')

# And now plot everything passing the basemap to the plotting functions
ft.vis.figure()
bm.drawcoastlines()
bm.drawmapboundary()
ft.vis.contourf(lon, lat, data, shape, 12, basemap=bm)
ft.vis.colorbar()
ft.vis.show()

