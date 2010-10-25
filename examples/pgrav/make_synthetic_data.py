"""
Make some synthetic FTG data.
"""

import pickle

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.grav import synthetic, io
import fatiando.utils
import fatiando.vis

# Get a logger for the script
log = fatiando.utils.get_logger()

prisms = []
prisms.append({'x1':-600, 'x2':-200, 'y1':-600, 'y2':-200, 'z1':200, 'z2':600,
               'value':1000})
prisms.append({'x1':000, 'x2':400, 'y1':-600, 'y2':-200, 'z1':200, 'z2':600,
               'value':700})
prisms.append({'x1':-400, 'x2':400, 'y1':200, 'y2':600, 'z1':200, 'z2':600,
               'value':500})

prisms = numpy.array(prisms)

fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)
dataset = fatiando.vis.plot_prism_mesh(prisms, style='surface', 
                                       label='Density kg/cm^3')
axes = mlab.axes(dataset, nb_labels=5, extent=[-800,800,-800,800,0,800])
mlab.show()

modelfile = open('model.pickle', 'w') 
pickle.dump(prisms, modelfile)
modelfile.close()

error = 1

pylab.figure(figsize=(16,8))
pylab.suptitle(r'Synthetic FTG data with %g $E\"otv\"os$ noise' 
               % (error), fontsize=16)

for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):

    data = synthetic.from_prisms(prisms, x1=-800, x2=800, y1=-800, y2=800, 
                                 nx=50, ny=50, height=150, field=field)
    
    data['value'], error = fatiando.utils.contaminate(data['value'], 
                                                      stddev=error, 
                                                      percent=False, 
                                                      return_stddev=True)
    
    data['error'] = error*numpy.ones(len(data['value']))

    io.dump('%s_data.txt' % (field), data)
    
    pylab.subplot(2, 3, i + 1)
    pylab.axis('scaled')
    pylab.title(field)
    X, Y, V = fatiando.utils.extract_matrices(data)
    pylab.contourf(X, Y, V, 10, cmap=pylab.cm.jet)
    cb = pylab.colorbar()
    cb.set_label(r'$E\"otv\"os$')
    
    pylab.xlim(data['x'].min(), data['x'].max())
    pylab.ylim(data['y'].min(), data['y'].max())

pylab.savefig("data.png")
pylab.show()