"""
Vis: Plot the Earth, continents, inner and outer core in 3D with Mayavi2
"""
from fatiando.vis import myv

myv.figure(zdown=False)
myv.continents(linewidth=2)
myv.earth(opacity=0.5)
myv.core(opacity=0.7)
myv.core(inner=True)
myv.show()
