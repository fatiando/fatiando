"""
Example of how to generate a SquareMesh and get the physical properties from an
image file.
"""
from os import path
from matplotlib import pyplot
from fatiando.mesher.dd import SquareMesh
from fatiando import vis, logger

log = logger.get()
log.info(logger.header())
log.info(__doc__)

imgfile = path.join(path.dirname(path.abspath(__file__)), 'fat-logo.png')
mesh = SquareMesh((0, 5000, 0, 5000), (150, 150))
mesh.img2prop(imgfile, 5, 10, 'slowness')

pyplot.figure()
pyplot.title('Slowness model of the Earth')
vis.map.squaremesh(mesh, mesh.props['slowness'])
cb = pyplot.colorbar()
cb.set_label("Slowness")
pyplot.show()
