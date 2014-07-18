"""
Vis: Plot a map using the Orthographic map projection and filled contours
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate some data to plot
area = (-40, 0, 10, -50)
shape = (100, 100)
lon, lat = gridder.regular(area, shape)
data = utils.gaussian2d(lon, lat, 10, 20, -20, -20, angle=-45)

# Now get a basemap to plot with some projection
bm = mpl.basemap(area, 'ortho')

# And now plot everything passing the basemap to the plotting functions
mpl.figure()
bm.bluemarble()
mpl.contourf(lon, lat, data, shape, 12, basemap=bm)
mpl.colorbar()
mpl.show()
