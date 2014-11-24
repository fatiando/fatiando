"""
Vis: Plot a map using the Mercator map projection and pseudo-color
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate some data to plot
area = (-20, 40, 20, 80)
shape = (100, 100)
lon, lat = gridder.regular(area, shape)
data = utils.gaussian2d(lon, lat, 10, 20, 10, 60, angle=45)

# Now get a basemap to plot with some projection
bm = mpl.basemap(area, 'merc')

# And now plot everything passing the basemap to the plotting functions
mpl.figure(figsize=(5, 8))
mpl.pcolor(lon, lat, data, shape, basemap=bm)
mpl.colorbar()
bm.drawcoastlines()
bm.drawmapboundary()
bm.drawcountries()
mpl.draw_geolines(area, 10, 10, bm)
mpl.show()
