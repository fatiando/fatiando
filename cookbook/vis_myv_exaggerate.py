"""
Vis: Exaggerate the vertical dimension of 3D plots
"""
from fatiando.mesher import Prism, PolygonalPrism
from fatiando.vis import myv

prism = Prism(0, 1000, 0, 1000, 0, 10)
poly = PolygonalPrism([[-2000, -2000], [-1000, -1000], [-2000, -1000]], 0, 10)
bounds = (-3000, 3000, -3000, 3000, 0, 20)
myv.figure()
myv.prisms([prism])
myv.polyprisms([poly])
myv.axes(myv.outline(bounds))
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.title('No exaggeration')
scale = (1, 1, 50)  # Exaggerate 50x the z axis
myv.figure()
myv.prisms([prism], scale=scale)
myv.polyprisms([poly], scale=scale)
myv.axes(myv.outline(bounds, scale=scale), ranges=bounds)
myv.wall_north(bounds, scale=scale)
myv.wall_bottom(bounds, scale=scale)
myv.title('50x exaggeration')
myv.show()
