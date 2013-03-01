"""
Meshing: Make and plot a tesseroid mesh
"""
from fatiando import mesher
from fatiando.vis import myv

mesh = mesher.TesseroidMesh((-60, 60, -30, 30, 100000, -500000), (10, 10, 10))

myv.figure(zdown=False)
myv.tesseroids(mesh)
myv.earth(opacity=0.3)
myv.continents()
myv.meridians(range(-180, 180, 30))
myv.parallels(range(-90, 90, 30))
myv.show()
