"""
Meshing: Generate a SquareMesh and get the physical properties from an image
"""
import urllib
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Make a square mesh
mesh = ft.mesher.SquareMesh((0, 5000, 0, 5000), (150, 150))
# Fetch the image from the online docs
urllib.urlretrieve(
    'http://fatiando.readthedocs.org/en/latest/_static/logo.png', 'logo.png')
# Load the image as the P wave velocity (vp) property of the mesh
mesh.img2prop('logo.png', 5000, 10000, 'vp')

ft.vis.figure()
ft.vis.title('P wave velocity model of the Earth')
ft.vis.squaremesh(mesh, prop='vp')
cb = ft.vis.colorbar()
cb.set_label("Vp (km/s)")
ft.vis.show()
