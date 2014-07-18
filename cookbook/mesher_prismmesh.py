"""
Meshing: Make and plot a 3D prism mesh
"""
from fatiando import mesher
from fatiando.vis import myv

mesh = mesher.PrismMesh(bounds=(-2, 2, -3, 3, 0, 1), shape=(4, 4, 4))

myv.figure()
plot = myv.prisms(mesh)
axes = myv.axes(plot)
myv.show()
