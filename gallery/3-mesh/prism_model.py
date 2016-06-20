"""
Simple prism model
------------------

Create a simple geologic model using rectangular prisms
(:class:`fatiando.mesher.Prism`) and plot it in 3D with
:mod:`fatiando.vis.myv`.

"""
from fatiando.mesher import Prism
from fatiando.vis import myv

# Models in Fatiando are basically sequences of geometric elements from
# fatiando.mesher
# Each element can have as many physical properties as you want.
# Here, we'll give the prisms a density and a magnetization vector.
model = [
    Prism(-1500, -500, -1500, -500, 1000, 2000,
          {'magnetization': [3, 5, 4], 'density': 500}),
    Prism(500, 1500, 1000, 2000, 500, 1500,
          {'magnetization': [10, 2, 1], 'density': -250}),
    ]

# Create a 3D figure in Mayavi
myv.figure()
# Give it some prisms to plot. The color will be decided by the density.
myv.prisms(model, prop='density')
# It's useful to plot axes and a bounding box about the prism
bounds = [-2500, 2500, -2500, 2500, 0, 2500]
myv.axes(myv.outline(bounds))
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.show()
