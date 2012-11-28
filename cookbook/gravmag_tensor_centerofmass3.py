"""
GravMag: Center of mass estimation using the first eigenvector of the gravity
gradient tensor (pyramid model)
"""
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Generate some synthetic data
prisms = [ft.mesher.Prism(-500,500,-500,500,500,1000,{'density':1000}),
          ft.mesher.Prism(-1000,1000,-1000,1000,1000,1500,{'density':1000}),
          ft.mesher.Prism(-2000,2000,-2000,2000,1500,2000,{'density':1000})]
shape = (100, 100)
xp, yp, zp = ft.gridder.regular((-5000, 5000, -5000, 5000), shape, z=-150)
noise = 1
tensor = [ft.utils.contaminate(ft.gravmag.prism.gxx(xp, yp, zp, prisms), noise),
          ft.utils.contaminate(ft.gravmag.prism.gxy(xp, yp, zp, prisms), noise),
          ft.utils.contaminate(ft.gravmag.prism.gxz(xp, yp, zp, prisms), noise),
          ft.utils.contaminate(ft.gravmag.prism.gyy(xp, yp, zp, prisms), noise),
          ft.utils.contaminate(ft.gravmag.prism.gyz(xp, yp, zp, prisms), noise),
          ft.utils.contaminate(ft.gravmag.prism.gzz(xp, yp, zp, prisms), noise)]
# Plot the data
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
ft.vis.figure()
for i, title in enumerate(titles):
    ft.vis.subplot(3, 2, i + 1)
    ft.vis.title(title)
    ft.vis.axis('scaled')
    levels = ft.vis.contourf(yp, xp, tensor[i], shape, 10)
    ft.vis.contour(yp, xp, tensor[i], shape, levels)
    ft.vis.m2km()
ft.vis.show()
# Get the eigenvectors from the tensor data
eigenvals, eigenvecs = ft.gravmag.tensor.eigen(tensor)
# Use the first eigenvector to estimate the center of mass
cm, sigma = ft.gravmag.tensor.center_of_mass(xp, yp, zp, eigenvecs[0])
print "Sigma = %g" % (sigma)
# Plot the prism and the estimated center of mass
ft.vis.figure3d()
ft.vis.points3d([cm], size=300.)
ft.vis.prisms(prisms, prop='density', opacity=0.5)
axes = ft.vis.axes3d(
    ft.vis.outline3d(extent=[-5000, 5000, -5000, 5000, 0, 5000]))
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
