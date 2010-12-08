"""
Generate synthetic vertical gravity data and upward continue it.
"""

import pylab

import fatiando.grav.synthetic as synthetic
import fatiando.grav.transform as transform
from fatiando import grid, vis, utils, geometry

log = utils.get_logger()
log.info(utils.header())

# Make a sphere model and generate the synthetic data
spheres = []
spheres.append(geometry.sphere(xc=1000, yc=1000, zc=1000, radius=800,
                               props={'density':500}))
spheres.append(geometry.sphere(xc=4000, yc=1000, zc=500, radius=300,
                               props={'density':1000}))
spheres.append(geometry.sphere(xc=4000, yc=4000, zc=1500, radius=1000,
                               props={'density':800}))
spheres.append(geometry.sphere(xc=1000, yc=4000, zc=1000, radius=700,
                               props={'density':-800}))
spheres.append(geometry.sphere(xc=2500, yc=2500, zc=1200, radius=1000,
                               props={'density':1200}))

data = grid.regular(0, 5000, 0, 5000, 25, 25, 0)

synthetic.from_spheres(spheres, data)

data['value'] = utils.contaminate(data['value'], stddev=0.2, percent=False)

# Calculate the data at the new height
new_height = 3000

updata_true = grid.regular(0, 5000, 0, 5000, 25, 25, -new_height)

synthetic.from_spheres(spheres, updata_true)

# Upward continue the data to the new height
updata = transform.upcontinue(data, new_height)

# Plot the results
pylab.figure(figsize=(16,8))

# Original data
pylab.subplot(1,2,1)
pylab.axis('scaled')
pylab.title("Data at 0 m")
levels = vis.contourf(data, 15)
vis.contour(data, levels)

# The analytical data at the new height
pylab.subplot(1,2,2)
pylab.axis('scaled')
pylab.title("Upward continued to %g m" % (new_height))
levels = vis.contour(updata_true, 10, color='b', label="Analytical")
vis.contour(updata, levels, color='r', label='Calculated')
pylab.legend(loc='lower right', shadow=True)

pylab.show()
