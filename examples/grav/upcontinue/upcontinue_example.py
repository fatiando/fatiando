"""
Generate synthetic vertical gravity data from a prism model and upward continue
it. 
"""

import pylab

import fatiando.grav.synthetic as synthetic
import fatiando.grav.transform as transform
import fatiando.utils as utils
import fatiando.vis as vis
import fatiando.grid

log = utils.get_logger()
log.info(utils.header())

# Define a prism as the model to generate the gravity data
prism = {'x1':-100, 'x2':100, 'y1':-100, 'y2':100, 'z1':500, 'z2':700,
         'value':1000}

# Generate the data at 0 height and at the new height
data = synthetic.from_prisms([prism], x1=-500, x2=500, y1=-500, y2=500, 
                             nx=25, ny=25, height=0, field='gz')

new_height = 42

updata_true = synthetic.from_prisms([prism], x1=-500, x2=500, y1=-500, y2=500, 
                                    nx=25, ny=25, height=new_height, field='gz')

# Upward continue the data
updata = transform.upcontinue(data, new_height)

# Calculate the difference in a percentage
diff = fatiando.grid.subtract(updata_true, updata, percent=True)

# Plot the results
pylab.figure(figsize=(12,8))

# Original data
pylab.subplot(2,2,1)
pylab.axis('scaled')
pylab.title("Original")
vis.pcolor(data)
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

# The analytical data at the new height
pylab.subplot(2,2,2)
pylab.axis('scaled')
pylab.title("Analytical")
vis.pcolor(updata_true)
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

# The difference
pylab.subplot(2,2,3)
pylab.axis('scaled')
pylab.title("Difference")
vis.pcolor(diff)
cb = pylab.colorbar()
cb.set_label("%")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

# The upward continued
pylab.subplot(2,2,4)
pylab.axis('scaled')
pylab.title("Upward continued")
vis.pcolor(updata, vmin=updata_true['value'].min(),
           vmax=updata_true['value'].max())
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

pylab.show()
