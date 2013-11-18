"""
Meshing: Make and plot a tesseroid with the Earth
"""
from fatiando import mesher
from fatiando.vis import myv

model = mesher.Tesseroid(-10, 10, 50, 60, 500000, 0)
myv.figure(zdown=False)
myv.tesseroids([model])
myv.earth()
myv.continents()
myv.meridians(range(-180, 180, 10))
myv.parallels(range(-90, 90, 10))
myv.show()
