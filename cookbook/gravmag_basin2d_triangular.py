"""
GravMag: Simple gravity inversion for the relief of a 2D triangular basin
"""
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
verts = [(10000, 1.), (90000, 1.), (80000, 5000)]
model = ft.mesher.Polygon(verts, {'density':-100})
xp = numpy.arange(0., 100000., 1000.)
zp = numpy.zeros_like(xp)
gz = ft.utils.contaminate(ft.gravmag.talwani.gz(xp, zp, [model]), 1)

log.info("Preparing for the inversion")
solver = ft.inversion.gradient.levmarq(initial=(10000, 1000))
estimate, residuals = ft.gravmag.basin2d.triangular(xp, zp, gz, verts[0:2], -100,
    solver)

ft.vis.figure()
ft.vis.subplot(2, 1, 1)
ft.vis.title("Gravity anomaly")
ft.vis.plot(xp, gz, 'ok', label='Observed')
ft.vis.plot(xp, gz - residuals, '-r', linewidth=2, label='Predicted')
ft.vis.legend(loc='lower left')
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
