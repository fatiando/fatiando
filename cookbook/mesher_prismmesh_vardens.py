"""
Meshing: Make a 3D prism mesh with depth-varying density
"""
from fatiando import gridder, mesher
from fatiando.vis import myv

shape = (10, 20, 10)
nz, ny, nx = shape
mesh = mesher.PrismMesh((0, 100, 0, 200, 0, 50), shape)


def fill(i):
    k = i / (nx * ny)
    return k
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

myv.figure()
myv.prisms(mesh, prop='density')
myv.axes(myv.outline(), fmt='%.0f')
myv.show()
