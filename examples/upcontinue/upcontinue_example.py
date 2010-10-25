"""
Generate synthetic vertical gravity data from a prism model and upward continue
it. 
"""

import pylab

import fatiando.grav.synthetic as synthetic
import fatiando.grav.transform as transform
import fatiando.utils as utils

log = utils.get_logger()

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

# Extract the matrices in order to plot the data
X, Y, Zdata = utils.extract_matrices(data)
X, Y, Zuptrue = utils.extract_matrices(updata_true)
X, U, Zup = utils.extract_matrices(updata)

# Also plot the difference between the analytical and the upward continued
Zdiff = abs(100*(Zuptrue - Zup)/Zuptrue)

# Plot the results
pylab.figure(figsize=(12,8))

# Original data
pylab.subplot(2,2,1)
pylab.axis('scaled')
pylab.title("Original")
pylab.pcolor(X, Y, Zdata, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

# The analytical data at the new height
pylab.subplot(2,2,2)
pylab.axis('scaled')
pylab.title("Analytical")
pylab.pcolor(X, Y, Zuptrue, cmap=pylab.cm.jet, vmin=Zuptrue.min(), 
             vmax=Zuptrue.max())
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

# The difference
pylab.subplot(2,2,3)
pylab.axis('scaled')
pylab.title("Difference")
pylab.pcolor(X, Y, Zdiff, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("%")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

# The upward continued
pylab.subplot(2,2,4)
pylab.axis('scaled')
pylab.title("Upward continued")
pylab.pcolor(X, Y, Zup, cmap=pylab.cm.jet, vmin=Zuptrue.min(), 
             vmax=Zuptrue.max())
cb = pylab.colorbar()
cb.set_label("mGal")
pylab.xlim(X.min(), X.max())
pylab.ylim(Y.min(), Y.max())

pylab.show()
