"""
Vis: Plot a map using the Robinson map projection and contours
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate some data to plot
area = (-180, 180, -80, 80)
shape = (100, 100)
lon, lat = gridder.regular(area, shape)
data = utils.gaussian2d(lon, lat, 30, 60, 10, 30, angle=-60)

# Now get a basemap to plot with some projection
bm = mpl.basemap(area, 'robin')

# And now plot everything passing the basemap to the plotting functions
mpl.figure()
mpl.contour(lon, lat, data, shape, 15, basemap=bm)
bm.drawcoastlines()
bm.drawmapboundary(fill_color='aqua')
bm.drawcountries()
bm.fillcontinents(color='coral')
mpl.draw_geolines((-180, 180, -90, 90), 60, 30, bm)
mpl.show()
