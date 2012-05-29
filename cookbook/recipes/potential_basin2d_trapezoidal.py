"""
Gravity inversion for the relief of a 2D trapezoidal basin
"""
import numpy
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Generating synthetic data")
verts = [(10000, 1.), (90000, 1.), (90000, 7000), (10000, 3330)]
model = ft.msh.dd.Polygon(verts, {'density':-100})
xp = numpy.arange(0., 100000., 1000.)
zp = numpy.zeros_like(xp)
gz = ft.utils.contaminate(ft.pot.talwani.gz(xp, zp, [model]), 0.5)

log.info("Preparing for the inversion")
dm = ft.pot.basin2d.TrapezoidalGzDM(xp, zp, gz, prop=-100, verts=verts[0:2])
solver = ft.inversion.gradient.levmarq(initial=(9000, 500))
p, residuals = ft.pot.basin2d.trapezoidal([dm], solver)
estimate = ft.msh.dd.Polygon([(10000, 1.), (90000, 1.), (90000, p[0]), (10000, p[1])])

ft.vis.figure()
ft.vis.subplot(2, 1, 1)
ft.vis.title("Gravity anomaly")
ft.vis.plot(xp, gz, 'ok', label='Observed')
ft.vis.plot(xp, gz - residuals[0], '-r', linewidth=2, label='Predicted')
ft.vis.legend(loc='lower left', numpoints=1)
ft.vis.ylabel("mGal")
ft.vis.xlim(0, 100000)
ft.vis.subplot(2, 1, 2)
ft.vis.polygon(estimate, 'o-r', linewidth=2, fill='r', alpha=0.3,
                label='Estimated')
ft.vis.polygon(model, '--k', linewidth=2, label='True')
ft.vis.legend(loc='lower left', numpoints=1)
ft.vis.xlabel("X")
ft.vis.ylabel("Z")
ft.vis.set_area((0, 100000, 10000, -500))
ft.vis.show()
