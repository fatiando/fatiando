"""
Make some synthetic FTG data.
"""


import pickle
import logging
log = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter())
log.addHandler(handler)
log.setLevel(logging.DEBUG)

import pylab
import numpy
from enthought.mayavi import mlab

from fatiando.gravity import synthetic, io
import fatiando.utils
from fatiando.visualization import plot_prism_mesh

prisms = []
prisms.append({'x1':-200, 'x2':200, 'y1':-200, 'y2':200, 'z1':000, 'z2':400,
               'value':1000})

prisms = numpy.array(prisms)

fig = mlab.figure()
fig.scene.background = (0.1, 0.1, 0.1)
fig.scene.camera.pitch(180)
fig.scene.camera.roll(180)
dataset = plot_prism_mesh(prisms, style='surface', label='Density kg/cm^3')
axes = mlab.axes(dataset, nb_labels=5, extent=[-800,800,-800,800,0,1600])
mlab.show()

modelfile = open('model.pickle', 'w') 
pickle.dump(prisms, modelfile)
modelfile.close()

error = 5

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

    io.dump('%s_data.txt' % (field), data)
    
    pylab.subplot(2, 3, i + 1)
    pylab.axis('scaled')
    
    X, Y, V = fatiando.utils.extract_matrices(data)
    pylab.contourf(X, Y, V, 10, cmap=pylab.cm.jet)
    cb = pylab.colorbar()
    cb.set_label(r'$E\"otv\"os$')
    
    pylab.xlim(data['x'].min(), data['x'].max())
    pylab.ylim(data['y'].min(), data['y'].max())

pylab.show()