"""
Make some synthetic FTG data.
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.grav import synthetic, io
from fatiando import utils, stats, vis


# Get a logger for the script
log = utils.get_logger()

# Set logging to a file
utils.set_logfile('make_data.log')

# Log a header with the current version info
log.info(utils.header())

# Make the prism model
prisms = []
prisms.append({'x1':1000, 'x2':2000, 'y1':1000, 'y2':2000, 'z1':-350, 'z2':700,
               'value':1000})
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
                             nx=25, ny=25, height=600, field='gz')
    
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
                                 nx=25, ny=25, height=600, field=field)
    
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

log.info("Generating synthetic topography")

# Make some synthetic topography
topo = {'x':data['x'], 'y':data['y'], 'nx':data['nx'], 'ny':data['ny'], 
        'grid':True}
cov = [[1500.**2, 0],[0, 1000.**2]]
topo['h'] = 500*stats.gaussian2d(data['x'], data['y'], 1500., 1500., cov)

io.dump_topo('topo.txt', topo)

# ... and plot it as well
pylab.figure()
pylab.title("Topography")
pylab.axis('scaled')
vis.pcolor(topo, vkey='h')
cb = pylab.colorbar()
cb.set_label("Height (m)")

pylab.savefig("topo.png")

pylab.show()