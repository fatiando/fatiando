"""
Make synthetic travel time data based on an image model.
"""

import logging

# Configure the logging output to print to stderr
baselog = logging.getLogger()
stderrhandle = logging.StreamHandler()
stderrhandle.setFormatter(logging.Formatter())
baselog.addHandler(stderrhandle)
baselog.setLevel(logging.DEBUG)

# Make a logger for the script
log = logging.getLogger('script')
log.setLevel(logging.DEBUG)

import pylab
import numpy

from fatiando.seismo import synthetic, io                           
import fatiando.utils
from fatiando.visualization import plot_src_rec, \
                                   plot_ray_coverage
                                   
# Load the image model and convert it to a velocity model
model = synthetic.vel_from_image('model.jpg', vmax=5, vmin=1)

# Shoot rays through the model simulating an X-ray tomography configuration
data = synthetic.shoot_cartesian_straight(model, src_n=20, rec_n=10, 
                                          type='xray', rec_span=45.)

# Contaminate the data with Gaussian noise
data['traveltime'], error = fatiando.utils.contaminate(data['traveltime'], 
                                                       stddev=0.005, 
                                                       percent=True, 
                                                       return_stddev=True)

log.info("Contaminated the data with %g noise" % (error))

data['error'] = error*numpy.ones(len(data['traveltime']))

# Save the output to a file
io.dump_traveltime('traveltimedata.txt', data)

# Visualize the results


# PLOT THE SYNTHETIC DATA AND INVERSION RESULTS
pylab.figure(figsize=(16,4))
pylab.suptitle("X-ray simulation: Synthetic data", fontsize=14)

pylab.subplot(1,3,1)
pylab.axis('scaled')
pylab.title("Synthetic velocity model")
ax = pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.subplot(1,3,2)
pylab.axis('scaled')
pylab.title("Source and receiver locations")
pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
plot_src_rec(data['src'], data['rec'], markersize=6)
pylab.xlim(-1.2*model.shape[1], 2.2*model.shape[1])
pylab.ylim(-1.2*model.shape[0], 2.2*model.shape[0])

pylab.subplot(1,3,3)
pylab.axis('scaled')
pylab.title("Ray coverage")
pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
plot_ray_coverage(data['src'], data['rec'], '-k')
plot_src_rec(data['src'], data['rec'], markersize=6)
pylab.xlim(-1.2*model.shape[1], 2.2*model.shape[1])
pylab.ylim(-1.2*model.shape[0], 2.2*model.shape[0])

pylab.show()