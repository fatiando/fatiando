"""
Make some synthetic FTG data.
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

import fatiando.grav.synthetic as synthetic
import fatiando.grav.io as io
import fatiando.utils as utils
import fatiando.vis as vis

# Get a logger for the script
log = utils.get_logger()

# Set logging to a file
utils.set_logfile('make_data.log')

# Log a header with the current version info
log.info(utils.header())

# Make the prism model
prisms = []
# Lower right
prisms.append({'x1':2000, 'x2':2700, 'y1':300, 'y2':1000, 'z1':000, 'z2':600,
               'value':1000})
# Long north-south
prisms.append({'x1':200, 'x2':700, 'y1':200, 'y2':2000, 'z1':500, 'z2':1000,
               'value':1500})
# Deep
prisms.append({'x1':1000, 'x2':1700, 'y1':1000, 'y2':1500, 'z1':800, 'z2':1800,
               'value':1500})
# Long east-west
prisms.append({'x1':400, 'x2':2600, 'y1':2300, 'y2':2800, 'z1':200, 'z2':700,
               'value':-1000})

prisms = numpy.array(prisms)

# Show the model before calculating to make sure it's right
fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
dataset = vis.plot_prism_mesh(prisms, style='surface', label='Density kg/cm^3')
axes = mlab.axes(dataset, nb_labels=5, extent=[0,3000,0,3000,-2000,0])
mlab.show()

# Pickle the model so that it can be shown next to the inversion result later
modelfile = open('model.pickle', 'w') 
pickle.dump(prisms, modelfile)
modelfile.close()

# Calculate the vertical gravitational effect
error_gz = 0.1
data = synthetic.from_prisms(prisms, x1=0, x2=3000, y1=0, y2=3000, 
                             nx=30, ny=30, height=150, field='gz')
    
data['value'] = utils.contaminate(data['value'], stddev=error_gz, 
                                  percent=False, return_stddev=False)

data['error'] = error_gz*numpy.ones(len(data['value']))

# ... save it
io.dump('gz_data.txt', data)

# ... and plot it
pylab.figure()
pylab.axis('scaled')
pylab.title(r"Synthetic $g_z$ with %g mGal noise" % (error_gz))
vis.contourf(data, 10)
cb = pylab.colorbar()
cb.set_label('mGal')
pylab.xlim(data['x'].min(), data['x'].max())
pylab.ylim(data['y'].min(), data['y'].max())
pylab.savefig("data_gz.png")

# Now calculate all the components of the gradient tensor
error = 2

pylab.figure(figsize=(16,8))
pylab.suptitle(r'Synthetic FTG data with %g $E\"otv\"os$ noise' 
               % (error), fontsize=16)

for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):

    data = synthetic.from_prisms(prisms, x1=0, x2=3000, y1=0, y2=3000, 
                                 nx=30, ny=30, height=150, field=field)
    
    data['value'], error = utils.contaminate(data['value'], 
                                             stddev=error, 
                                             percent=False, 
                                             return_stddev=True)
    
    data['error'] = error*numpy.ones(len(data['value']))

    io.dump('%s_data.txt' % (field), data)
    
    pylab.subplot(2, 3, i + 1)
    pylab.axis('scaled')
    pylab.title(field)
    vis.contourf(data, 10)
    cb = pylab.colorbar()
    cb.set_label(r'$E\"otv\"os$')
    
    pylab.xlim(data['x'].min(), data['x'].max())
    pylab.ylim(data['y'].min(), data['y'].max())

pylab.savefig("data_ftg.png")

pylab.show()