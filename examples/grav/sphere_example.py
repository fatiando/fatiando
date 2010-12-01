"""
Generate synthetic data using a sphere model.
"""

import pylab

from fatiando.grav import synthetic, io
from fatiando import utils, vis, geometry, grid

# Set logging to default (stderr)
log = utils.get_logger()

log.info(utils.header())

# Make the sphere model
spheres = []
spheres.append(geometry.sphere(xc=500, yc=500, zc=500, radius=250,
                               props={'density':1000}))

# Need to make a grid to calculate gz on
data = grid.regular(x1=0, x2=1000, y1=0, y2=1000, nx=100, ny=100, z=0)

# Now we can calculate gz caused by 'spheres' on 'grid'
synthetic.from_spheres(spheres, data, field='gz')

# Finally, plot the calculated data and save it
io.dump('sphere_gz_data.txt', data)

pylab.figure()
pylab.axis('scaled')
vis.pcolor(data)
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.show()