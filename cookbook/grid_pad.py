"""
Gridding: Pad gridded data
"""

from fatiando import mesher,gridder
from fatiando.gravmag import prism
from fatiando.vis import mpl
import numpy as np

# Generate gridded data
shape = (101,172)
x,y,z = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
model = [mesher.Prism(-4000, -3000, -4000, -3000, 0, 2000, {'density':1000}),
        mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000, {'density':-900}),
        mesher.Prism(2000, 4000, 3000, 4000, 0, 2000, {'density':1300})]
gz = prism.gz(x, y, z, model)
gz = gz.reshape(shape)

# Pad arrays with all the padding options
pads = []
xy = np.zeros((len(x),2))
xy[:,0] = x[:]
xy[:,1] = y[:]

# Pad with zeros (note padtype can be a string or numeric, so long as it
# can be cast as a float)
g,xyp,nps = gridder.pad_array(gz, xy, padtype='0')
pads.append(g.flatten())

# Pad with the mean of each vector
g,_,_ = gridder.pad_array(gz, xy, padtype='mean')
pads.append(g.flatten())

# Pad with the edge of each vector
g,_,_ = gridder.pad_array(gz, xy, padtype='edge')
pads.append(g.flatten())

# Pad with a linear taper
g,_,_ = gridder.pad_array(gz, xy, padtype='lintaper')
pads.append(g.flatten())

# Pad with the even reflection
g,_,_ = gridder.pad_array(gz, xy, padtype='reflection')
pads.append(g.flatten())

# Pad with the odd reflection
g,_,_ = gridder.pad_array(gz, xy, padtype='OddReflection')
pads.append(g.flatten())

# Pad with the odd reflection and a cosine taper (default)
g,_,_ = gridder.pad_array(gz, xy, padtype='OddReflectionTaper')
pads.append(g.flatten())

shapepad = g.shape

# Generate new meshgrid and plot results
[Yp,Xp] = np.meshgrid(xyp[1], xyp[0])
yp = Yp.ravel()
xp = Xp.ravel()
titles = ['Original', 'Zero', 'Mean', 'Edge', 'Linear Taper', 'Reflection',
        'Odd Reflection', 'Odd Reflection/Taper']
mpl.figure(figsize=(17,9))
mpl.suptitle('Padding algorithms for a 2D array')
for ii,p in enumerate(pads):
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

