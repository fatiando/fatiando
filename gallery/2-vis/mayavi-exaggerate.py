"""
Exaggerate a dimension in 3D plots
----------------------------------

Sometimes things are too small on one dimension to plot properly. On the Earth,
this is usually the vertical dimension. The functions in
:mod:`fatiando.vis.myv` have a ``scale`` attribute to control how much
exaggeration should  be placed in each dimension of your plot.

"""
import copy
from fatiando.vis import myv
from fatiando.mesher import Prism


# Make two objects that are very thin.
model = [Prism(0, 1000, 0, 1000, 0, 10, props={'density': 300}),
         Prism(-2500, -1000, -2000, -500, 0, 5, props={'density': -300})]

bounds = [-3000, 3000, -3000, 3000, 0, 20]

# The scale argument is by how much each dimension (being x, y, and z) will be
# multiplied. This means 300x in the z dimension.
scale = (1, 1, 300)

# Pass "scale" along to all plot functions
myv.figure()
myv.prisms(model, prop='density', scale=scale)
myv.axes(myv.outline(bounds, scale=scale), ranges=bounds)
myv.wall_north(bounds, scale=scale)
myv.wall_bottom(bounds, scale=scale)
# Note: the tittle can't be the first thing on the plot.
myv.title('{}x vertical exaggeration'.format(scale[-1]))
myv.show()
