import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())

bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    ft.msh.ddd.Prism(-1500, -500, -500, 500, 1000, 2000, {'magnetization':2})]
shape = (100, 100)
area = bounds[0:4]
xp, yp, zp = ft.grd.regular(area, shape, z=-1)
baselevel = 10
tf = (ft.pot.prism.tf(xp, yp, zp, model, inc=-45, dec=0)/ft.constants.T2NT
      + baselevel)
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

point, base = ft.pot.euler.classic(xp, yp, zp, tf, xderiv, yderiv, zderiv, 3)
print "Base level used: %g" % (baselevel)
print "Estimated base level: %g" % (base)

ft.vis.figure3d()
ft.vis.points3d([point], size=300.)
ft.vis.prisms(model, prop='magnetization', opacity=0.5)
axes = ft.vis.axes3d(ft.vis.outline3d(extent=bounds))
ft.vis.wall_bottom(axes.axes.bounds, opacity=0.2)
ft.vis.wall_north(axes.axes.bounds)
ft.vis.show3d()
