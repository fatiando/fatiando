"""
Vis: Plot contour lines and filled contours
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate some data to plot
area = (-40, 0, 10, -50)
shape = (100, 100)
lon, lat = gridder.regular(area, shape)
data = utils.gaussian2d(lon, lat, 10, 20, -20, -20, angle=-45)

mpl.figure()

# Filled contour plot
mpl.subplot(221)
mpl.title('Filled contours')
mpl.contourf(lon, lat, data, shape, 50)
mpl.colorbar()

# Contour plot
mpl.subplot(222)
mpl.title('Line contours')
mpl.contour(lon, lat, data, shape, 10, color='r', style='dashed')

# Mix contour and contourf
# contour and contourf return a list of the contour values
mpl.subplot(223)
mpl.title('Filled contours + line contours')
levels = mpl.contourf(lon, lat, data, shape, 10)
mpl.colorbar()
# using "levels" tells contour to use the exact same values as contourf
mpl.contour(lon, lat, data, shape, levels)

# Limiting the scale of the data
mpl.subplot(224)
mpl.title('Using vmin and vmax')
mpl.contourf(lon, lat, data, shape, 50, vmin=0.5, vmax=0.8)
mpl.colorbar()

mpl.show()
