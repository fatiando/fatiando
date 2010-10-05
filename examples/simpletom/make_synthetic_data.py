"""
Make synthetic travel time data based on an image model.
"""

import pickle

import pylab
import numpy

from fatiando.seismo import synthetic, io                           
import fatiando.utils
import fatiando.vis
                    
# Make a logger for the script
log = fatiando.utils.get_logger()
               
# Load the image model and convert it to a velocity model
model = synthetic.vel_from_image('model.jpg', vmax=5., vmin=1.)

modelfile = open("model.pickle", 'w')
pickle.dump(model, modelfile)
modelfile.close()

# Shoot rays through the model simulating an X-ray tomography configuration
data = synthetic.shoot_cartesian_straight(model, src_n=10, rec_n=10, 
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
pylab.figure()
pylab.axis('scaled')
pylab.title("X-ray simulation: Synthetic velocity model")
ax = pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
pylab.xlim(0, model.shape[1])
pylab.ylim(0, model.shape[0])

pylab.figure()
pylab.axis('scaled')
pylab.title("X-ray simulation: Source and receiver locations")
pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
fatiando.vis.plot_src_rec(data['src'], data['rec'], markersize=10)
pylab.legend(numpoints=1, prop={'size':10}, shadow=True)
pylab.xlim(-1.2*model.shape[1], 2.2*model.shape[1])
pylab.ylim(-1.2*model.shape[0], 2.2*model.shape[0])

pylab.figure()
pylab.axis('scaled')
pylab.title("X-ray simulation: Ray coverage")
pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
fatiando.vis.plot_ray_coverage(data['src'], data['rec'], '-k')
fatiando.vis.plot_src_rec(data['src'], data['rec'], markersize=10)
pylab.legend(numpoints=1, prop={'size':10}, shadow=True)
pylab.xlim(-1.2*model.shape[1], 2.2*model.shape[1])
pylab.ylim(-1.2*model.shape[0], 2.2*model.shape[0])

pylab.show()