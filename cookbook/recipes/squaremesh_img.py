"""
Example of how to generate a SquareMesh and get the physical properties from an
image file.
"""
from os import path
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

imgfile = path.join(path.dirname(path.abspath(__file__)), 'fat-logo.png')
mesh = ft.msh.dd.SquareMesh((0, 5000, 0, 5000), (150, 150))
mesh.img2prop(imgfile, 5, 10, 'slowness')

ft.vis.figure()
ft.vis.title('Slowness model of the Earth')
ft.vis.squaremesh(mesh, prop='slowness')
cb = ft.vis.colorbar()
cb.set_label("Slowness")
ft.vis.show()
