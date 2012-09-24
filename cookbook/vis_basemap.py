"""
Vis: Plot a map using the Orthographic map projection and filled contours
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Generate some data to plot
area = (-40, 0, 10, -50)
shape = (100, 100)
lon, lat = ft.grd.regular(area, shape)
data = ft.utils.gaussian2d(lon, lat, 10, 20, -20, -20, angle=-45)

# Now get a basemap to plot with some projection
bm = ft.vis.basemap(area, 'ortho')

# And now plot everything passing the basemap to the plotting functions
ft.vis.figure()
bm.bluemarble()
ft.vis.contourf(lon, lat, data, shape, 12, basemap=bm)
ft.vis.colorbar()
ft.vis.show()

