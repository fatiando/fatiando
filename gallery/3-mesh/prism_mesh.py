"""
Regular prism mesh
--------------------

The mesh classes in Fatiando are more efficient ways of representing regular
meshes than simples lists of :class:`~fatiando.mesher.Prism` objects. This is
how you can create a :class:`~fatiando.mesher.PrismMesh` and assign it a
density for each prism.

"""
from __future__ import print_function
from fatiando.mesher import PrismMesh
from fatiando.vis import myv

mesh = PrismMesh(bounds=(0, 100, 0, 200, 0, 150), shape=(5, 6, 7))
# We'll give each prism a density value corresponding to it's index on the
# mesh. Notice that meshes take lists/arrays as their property values
mesh.addprop('density', list(range(mesh.size)))

# You can iterate over meshes like lists of elements
for p in mesh:
    print(p.props['density'], end=' ')

scene = myv.figure(size=(600, 600))
# Because you can iterate over a mesh, you can pass it anywhere a list of
# prisms is accepted
plot = myv.prisms(mesh, prop='density')
# The code below enables and configures the color bar
plot.module_manager.scalar_lut_manager.show_scalar_bar = True
plot.module_manager.scalar_lut_manager.lut_mode = 'Greens'
plot.module_manager.scalar_lut_manager.reverse_lut = True
plot.module_manager.scalar_lut_manager.label_text_property.color = (0, 0, 0)
plot.module_manager.scalar_lut_manager.title_text_property.color = (0, 0, 0)
plot.module_manager.scalar_lut_manager.scalar_bar_representation.position = [0.9, 0.4]
plot.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = [0.1, 0.6]

myv.axes(myv.outline(), fmt='%.1f')
myv.show()
