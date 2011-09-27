"""
Example of generating a prism mesh and extracting a submesh
"""
from enthought.mayavi import mlab
from fatiando import logger, vis
from fatiando.mesher.prism import Mesh3D, fill_mesh, vfilter, mesh2prisms, extract

vis.mlab = mlab

log = logger.get()
log.info(logger.header())
log.info("Example of generating a prism mesh, filling and cutting it")

shape = (5, 20, 10)
def fill(i):
    if i%2 == 0:
        return 1
    return -1
scalars = [fill(i) for i in xrange(shape[0]*shape[1]*shape[2])]
mesh = fill_mesh(scalars, Mesh3D(0, 100, 0, 200, 0, 50, shape))

odd = vfilter(-1, 0, 'density', mesh2prisms(mesh, prop='density'))
even = vfilter(0, 1, 'density', mesh2prisms(mesh, prop='density'))

mlab.figure()
vis.prisms3D(mesh2prisms(mesh), mesh['cells'])

mlab.figure()
vis.prisms3D(odd, extract('density', odd), style='wireframe')
vis.prisms3D(even, extract('density', even))

mlab.show()
