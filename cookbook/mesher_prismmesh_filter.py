"""
Meshing: Filter prisms from a 3D prism mesh based on their physical properties
"""
from fatiando import gridder, mesher
from fatiando.vis import myv

shape = (5, 20, 10)
bounds = (0, 100, 0, 200, 0, 50)
mesh = mesher.PrismMesh(bounds, shape)
# Fill the even prisms with 1 and odd with -1


def fill(i):
    if i % 2 == 0:
        return 1
    return -1
mesh.addprop('density', [fill(i) for i in xrange(mesh.size)])

# Separate even and odd prisms
odd = mesher.vfilter(-1, 0, 'density', mesh)
even = mesher.vfilter(0, 1, 'density', mesh)

myv.figure()
myv.prisms(odd, prop='density', vmin=-1, vmax=1)
myv.prisms(even, prop='density', style='wireframe', vmin=-1, vmax=1)
myv.axes(myv.outline(bounds))
myv.show()
