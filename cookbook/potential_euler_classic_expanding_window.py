"""
Potential: Classic 3D Euler deconvolution of magnetic data using an
expanding window
"""
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())

# Make a model
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    ft.msh.ddd.Prism(-1500, -500, -1500, -500, 1000, 2000, {'magnetization':2}),
    ft.msh.ddd.Prism(500, 1500, 500, 2000, 1000, 2000, {'magnetization':2})]
# Generate some data from the model
shape = (100, 100)
area = bounds[0:4]
xp, yp, zp = ft.gridder.regular(area, shape, z=-1)
# Add a constant baselevel
baselevel = 10
# Convert from nanoTesla to Tesla because euler and derivatives require things
# in SI
tf = (ft.utils.nt2si(ft.pot.prism.tf(xp, yp, zp, model, inc=-45, dec=0))
      + baselevel)
# Calculate the derivatives using FFT
xderiv = ft.pot.fourier.derivx(xp, yp, tf, shape)
yderiv = ft.pot.fourier.derivy(xp, yp, tf, shape)
zderiv = ft.pot.fourier.derivz(xp, yp, tf, shape)

ft.vis.figure()
titles = ['Total field', 'x derivative', 'y derivative', 'z derivative']
for i, f in enumerate([tf, xderiv, yderiv, zderiv]):
    ft.vis.subplot(2, 2, i + 1)
    ft.vis.title(titles[i])
    ft.vis.axis('scaled')
    ft.vis.contourf(yp, xp, f, shape, 50)
    ft.vis.colorbar()
    ft.vis.m2km()
ft.vis.show()

# Pick the center of the expanding window
ft.vis.figure()
ft.vis.suptitle('Pick the center of the expanding window')
ft.vis.axis('scaled')
ft.vis.contourf(yp, xp, tf, shape, 50)
ft.vis.colorbar()
center = ft.vis.map.pick_points(area, ft.vis.gca(), xy2ne=True)[0]

# Run the euler deconvolution on an expanding window
# Structural index is 3
index = 3
results = ft.pot.euler.expanding_window(xp, yp, zp, tf, xderiv, yderiv, zderiv,
    index, ft.pot.euler.classic, center, 500, 5000)
print "Base level used: %g" % (baselevel)
print "Estimated base level: %g" % (results['baselevel'])
print "Estimated source location: %s" % (str(results['point']))

ft.vis.figure3d()
ft.vis.points3d([results['point']], size=300.)
ft.vis.prisms(model, prop='magnetization', opacity=0.5)
axes = ft.vis.axes3d(ft.vis.outline3d(extent=bounds))
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
