import fatiando as ft
import numpy as np

log = ft.log.get()
log.info(ft.log.header())

bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    ft.msh.ddd.Prism(-1500, -500, -500, 500, 1000, 2000, {'magnetization':2})]
shape = (20, 20)
area = bounds[0:4]
xp, yp, zp = ft.grd.regular(area, shape, z=-150)
tf = ft.utils.contaminate(
        ft.pot.prism.tf(xp, yp, zp, model, inc=-45, dec=0)/ft.constants.T2NT,
        0.08, percent=True)
baselevel = 0.1*max(abs(tf))
tf += baselevel
xderiv = ft.pot.fourier.derivx(xp, yp, tf, shape)
yderiv = ft.pot.fourier.derivy(xp, yp, tf, shape)
zderiv = ft.pot.fourier.derivz(xp, yp, tf, shape)

ft.vis.figure()
titles = ['Total field', 'x derivative', 'y derivative', 'z derivative']
data = [tf, xderiv, yderiv, zderiv]
for i, f in enumerate(data):
    ft.vis.subplot(2, 2, i + 1)
    ft.vis.title(titles[i])
    ft.vis.axis('scaled')
    ft.vis.contourf(yp, xp, f, shape, 50)
    ft.vis.colorbar()
    ft.vis.m2km()
ft.vis.show()

# Run a standard Euler deconvolution to compare results
point_classic, base_classic = ft.pot.euler.classic(xp, yp, zp, tf, xderiv,
    yderiv, zderiv, 3)
    
results = ft.pot.euler.implicit(xp, yp, zp, tf, xderiv, yderiv, zderiv, 3)
point, base = results[:2]
print "Base level used: %g" % (baselevel)
print "Estimated base level (implicit): %g" % (base)
print "Estimated base level (classic): %g" % (base_classic)

print "Residuals stddev:"
ft.vis.figure(figsize=(14,6))
ft.vis.subplots_adjust(left=0.03, right=0.98)
ft.vis.suptitle("Observed + predicted data and histogram of residuals")
pred = results[2:]
for i in xrange(4):
    ft.vis.subplot(2, 4, 2*i + 1)
    ft.vis.title(titles[i])
    ft.vis.axis('scaled')
    levels = ft.vis.contourf(yp, xp, data[i], shape, 12)
    ft.vis.colorbar()
    ft.vis.contour(yp, xp, pred[i], shape, levels)
    ft.vis.m2km()
    print "  %s = %g" % (titles[i], np.std(data[i] - pred[i]))
    ft.vis.subplot(2, 4, 2*i + 2)
    ft.vis.title(titles[i])
    ft.vis.hist(data[i] - pred[i], 10)
ft.vis.show()

ft.vis.figure3d()
ft.vis.points3d([point], color=(0, 1, 0), size=250.)
ft.vis.points3d([point_classic], color=(1, 0, 0), size=250.)
ft.vis.prisms(model, prop='magnetization', opacity=0.5)
axes = ft.vis.axes3d(ft.vis.outline3d(extent=bounds),
    ranges=[0.001*i for i in bounds])
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
