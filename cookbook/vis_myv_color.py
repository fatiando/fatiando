"""
Vis: Set the colors in figures, prisms, polygonal prisms and tesseroids.
"""
from fatiando.mesher import Prism, PolygonalPrism, Tesseroid
from fatiando.vis import myv

prism = Prism(1, 2, 1, 2, 0, 1, {'density': 1})
polyprism = PolygonalPrism([[3, 1], [4, 2], [5, 1]], -1, 2, {'density': 2})
tesseroid = Tesseroid(10, 20, 50, 60, 10 ** 6, 0)

red, green, blue = (1, 0, 0), (0, 1, 0), (0, 0, 1)
white, black = (1, 1, 1), (0, 0, 0),

myv.figure()
# Make the prism red with blue edges, despite its density
myv.prisms([prism], 'density', color=red, edgecolor=blue)
# and the polyprism green with blue edges
myv.title('Body + edge colors')

myv.figure()
# For wireframes, color is usually set by the density.
# Overwrite this by setting *color*
# *edgecolor* is ignored
myv.polyprisms([polyprism], 'density', style='wireframe', color=green,
               edgecolor=red, linewidth=2)
myv.title('Wireframe colors')

# Black background, white lines, green tesseroid
myv.figure(zdown=False, color=black)
myv.earth()
myv.continents(color=white)
myv.tesseroids([tesseroid], color=green, edgecolor=white)
myv.title('Black background', color=white)
myv.show()
