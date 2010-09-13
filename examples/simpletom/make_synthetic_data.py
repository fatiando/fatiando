"""
Generate some synthetic travel time data using the square-model.jpg image.
"""

import logging
logging.basicConfig()

import pylab

from fatiando.seismo import synthetic, io
from fatiando.visualization import plot_src_rec, plot_ray_coverage

model = synthetic.vel_from_image('square-model.jpg', vmax=5, vmin=1)

data = synthetic.shoot_cartesian_straight(model, src_n=5, rec_n=10, type='xray')

io.dump_traveltime('traveltimes.txt', data)

pylab.figure(figsize=(14,5))

pylab.subplot(1,2,1)
pylab.axis('scaled')
pylab.title("Velocity model")
ax = pylab.pcolor(model, cmap=pylab.cm.jet)
cb = pylab.colorbar()
cb.set_label("Velocity")
plot_src_rec(data['src'], data['rec'])
pylab.xlim(-35, model.shape[1] + 35)
pylab.ylim(-35, model.shape[0] + 35)

pylab.subplot(1,2,2)
pylab.axis('scaled')
pylab.title("Ray coverage")
ax = pylab.pcolor(model, cmap=pylab.cm.jet)
plot_ray_coverage(data['src'], data['rec'], '-k')
plot_src_rec(data['src'], data['rec'])
pylab.xlim(-35, model.shape[1] + 35)
pylab.ylim(-35, model.shape[0] + 35)

pylab.show()