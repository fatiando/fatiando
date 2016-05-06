"""
Gridding: Pad gridded data
"""

from fatiando import mesher, gridder
from fatiando.gravmag import prism
from fatiando.vis import mpl
import numpy as np

# Generate gridded data
shape = (101, 172)
x, y, z = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
model = [mesher.Prism(-4000, -3000, -4000, -3000, 0, 2000, {'density': 1000}),
         mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density': -900}),
         mesher.Prism(2000, 4000, 3000, 4000, 0, 2000, {'density': 1300})]
gz = prism.gz(x, y, z, model)
gz = gz.reshape(shape)

# Pad arrays with all the padding options
pads = []
xy = [x, y]

# Pad with zeros, or any other number
# (note padtype can be a string or numeric, so long as it
# can be cast as a float.)  For example, both 0 and '0' are valid, as
# would be any other number.
g, nps = gridder.pad_array(gz, padtype='0')
pads.append(g.flatten())

# Pad with the mean of each vector
g, _ = gridder.pad_array(gz, padtype='mean')
pads.append(g.flatten())

# Pad with the edge of each vector
g, _ = gridder.pad_array(gz, padtype='edge')
pads.append(g.flatten())

# Pad with a linear taper
g, _ = gridder.pad_array(gz, padtype='lintaper')
pads.append(g.flatten())

# Pad with the even reflection
g, _ = gridder.pad_array(gz, padtype='reflection')
pads.append(g.flatten())

# Pad with the odd reflection
g, _ = gridder.pad_array(gz, padtype='OddReflection')
pads.append(g.flatten())

# Pad with the odd reflection and a cosine taper (default)
g, _ = gridder.pad_array(gz, padtype='OddReflectionTaper')
pads.append(g.flatten())

# Get coordinate vectors
N = gridder.pad_coords(xy, gz.shape, nps)

shapepad = g.shape

# Generate new meshgrid and plot results
yp = N[1]
xp = N[0]
titles = ['Original', 'Zero', 'Mean', 'Edge', 'Linear Taper', 'Reflection',
          'Odd Reflection', 'Odd Reflection/Taper']
mpl.figure(figsize=(17, 9))
mpl.suptitle('Padding algorithms for a 2D array')
for ii, p in enumerate(pads):
    mpl.subplot(2, 4, ii+2)
    mpl.axis('scaled')
    mpl.title(titles[ii+1])
    levels = mpl.contourf(yp*0.001, xp*0.001, p, shapepad, 15)
    cb = mpl.colorbar()
    mpl.contour(yp*0.001, xp*0.001, p, shapepad, levels, clabel=False,
                linewidth=0.1)
mpl.subplot(2, 4, 1)
mpl.axis('scaled')
mpl.title(titles[0])
levels = mpl.contourf(y*0.001, x*0.001, gz, shape, 15)
cb = mpl.colorbar()
mpl.contour(y*0.001, x*0.001, gz, shape, levels, clabel=False, linewidth=0.1)
mpl.show()
