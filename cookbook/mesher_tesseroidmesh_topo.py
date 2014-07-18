"""
Meshing: Make and plot a tesseroid mesh with topography
"""
from fatiando import gridder, utils, mesher
from fatiando.vis import myv

w, e = -2, 2
s, n = -2, 2
bounds = (w, e, s, n, 500000, 0)

x, y = gridder.regular((w, e, s, n), (50, 50))
height = (250000 +
          -100000 * utils.gaussian2d(x, y, 1, 5, x0=-1, y0=-1, angle=-60) +
          250000 * utils.gaussian2d(x, y, 1, 1, x0=0.8, y0=1.7))

mesh = mesher.TesseroidMesh(bounds, (20, 50, 50))
mesh.carvetopo(x, y, height)

scene = myv.figure(zdown=False)
myv.tesseroids(mesh)
myv.earth(opacity=0.3)
myv.continents()
scene.scene.camera.position = [
    21592740.078245595, 22628783.944262519, -28903782.916664094]
scene.scene.camera.focal_point = [
    5405474.2152075395, -1711034.715136874, 2155879.3486608281]
scene.scene.camera.view_angle = 1.6492674416639987
scene.scene.camera.view_up = [
    0.91713422625547714, -0.1284658947859818, 0.37730799740742887]
scene.scene.camera.clipping_range = [20169510.286021926, 69721043.718536735]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
myv.show()
