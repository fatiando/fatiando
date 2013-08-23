"""
Meshing: Generate a SquareMesh and get the physical properties from an image
"""
import urllib
from fatiando import mesher
from fatiando.vis import mpl

# Make a square mesh
mesh = mesher.SquareMesh((0, 5000, 0, 5000), (150, 150))
# Fetch the image from the online docs
urllib.urlretrieve(
    'http://fatiando.readthedocs.org/en/latest/_static/logo.png', 'logo.png')
# Load the image as the P wave velocity (vp) property of the mesh
mesh.img2prop('logo.png', 5000, 10000, 'vp')

mpl.figure()
mpl.title('P wave velocity model of the Earth')
mpl.squaremesh(mesh, prop='vp')
cb = mpl.colorbar()
cb.set_label("Vp (km/s)")
mpl.show()
