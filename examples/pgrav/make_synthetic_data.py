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
prisms.append({'x1':600, 'x2':1200, 'y1':200, 'y2':4200, 'z1':100, 'z2':600,
               'value':1300})
prisms.append({'x1':3000, 'x2':4000, 'y1':1000, 'y2':2000, 'z1':200, 'z2':800,
               'value':1000})
prisms.append({'x1':2700, 'x2':3200, 'y1':3700, 'y2':4200, 'z1':0, 'z2':900,
               'value':1200})
prisms.append({'x1':1500, 'x2':4500, 'y1':2500, 'y2':3000, 'z1':100, 'z2':500,
               'value':1500})

prisms = numpy.array(prisms)

fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)
dataset = fatiando.vis.plot_prism_mesh(prisms, style='surface', 
                                       label='Density kg/cm^3')
axes = mlab.axes(dataset, nb_labels=5, extent=[0,5000,0,5000,0,1000])
mlab.show()

modelfile = open('model.pickle', 'w') 
pickle.dump(prisms, modelfile)
modelfile.close()

error_gz = 0.1
data = synthetic.from_prisms(prisms, x1=0, x2=5000, y1=0, y2=5000, 
                             nx=50, ny=50, height=150, field='gz')
    
data['value'] = fatiando.utils.contaminate(data['value'], stddev=error_gz, 
                                           percent=False, return_stddev=False)

data['error'] = error_gz*numpy.ones(len(data['value']))

io.dump('gz_data.txt', data)

pylab.figure()
pylab.axis('scaled')
pylab.title(r"Synthetic $g_z$ with %g mGal noise" % (error_gz))
X, Y, V = fatiando.utils.extract_matrices(data)
pylab.contourf(X, Y, V, 10, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label('mGal')
pylab.xlim(data['x'].min(), data['x'].max())
pylab.ylim(data['y'].min(), data['y'].max())
pylab.savefig("data-gz.png")

error = 2

pylab.figure(figsize=(16,8))
pylab.suptitle(r'Synthetic FTG data with %g $E\"otv\"os$ noise' 
               % (error), fontsize=16)

for i, field in enumerate(['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']):

    data = synthetic.from_prisms(prisms, x1=0, x2=5000, y1=0, y2=5000, 
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