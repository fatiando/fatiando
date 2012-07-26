"""
Estimate the center of mass of a source using the eigenvectors of the gravity
gradient tensor.
"""
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

# Generate some synthetic data
prisms = [ft.msh.ddd.Prism(-1000,1000,-1000,1000,1000,3000,{'density':1000})]
shape = (100, 100)
xp, yp, zp = ft.grd.regular((-5000, 5000, -5000, 5000), shape, z=-150)
tensor = [ft.pot.prism.gxx(xp, yp, zp, prisms),
          ft.pot.prism.gxy(xp, yp, zp, prisms),
          ft.pot.prism.gxz(xp, yp, zp, prisms),
          ft.pot.prism.gyy(xp, yp, zp, prisms),
          ft.pot.prism.gyz(xp, yp, zp, prisms),
          ft.pot.prism.gzz(xp, yp, zp, prisms)]
# Get the eigenvectors from the tensor data
eigenvals, eigenvecs = ft.pot.tensor.eigen(tensor)
# Use the first eigenvector to estimate the center of mass
cm, sigma = ft.pot.tensor.center_of_mass(xp, yp, zp, eigenvecs[0])
# Plot the prism and the estimated center of mass
ft.vis.figure3d()
ft.vis.points3d([cm], size=200.)
ft.vis.prisms(prisms, prop='density', opacity=0.5)
axes = ft.vis.axes3d(
    ft.vis.outline3d(extent=[-5000, 5000, -5000, 5000, 0, 5000]))
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
