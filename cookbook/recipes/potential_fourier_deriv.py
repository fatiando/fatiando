"""
Calculate the derivative of the gravity anomaly in the x-direction
"""
import fatiando as ft

from fatiando.constants import SI2MGAL, SI2EOTVOS

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

log.info("Generating synthetic data")
prisms = [ft.msh.ddd.Prism(-1000,1000,-1000,1000,0,2000,{'density':100})]
area = (-5000, 5000, -5000, 5000)
shape = (51, 51)
z0 = -500
xp, yp, zp = ft.grd.regular(area, shape, z=z0)
gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, prisms), 0.001)

log.info("Calculating the x-derivative")
gxz = ft.pot.fourier.derivx(xp, yp, gz/SI2MGAL, shape)*SI2EOTVOS
gyz = ft.pot.fourier.derivy(xp, yp, gz/SI2MGAL, shape)*SI2EOTVOS
gzz = ft.pot.fourier.derivz(xp, yp, gz/SI2MGAL, shape)*SI2EOTVOS

log.info("Computing true values of the derivative")
gxz_true = ft.pot.prism.gxz(xp, yp, zp, prisms)
gyz_true = ft.pot.prism.gyz(xp, yp, zp, prisms)
gzz_true = ft.pot.prism.gzz(xp, yp, zp, prisms)

log.info("Plotting")
ft.vis.figure()
ft.vis.title("Original gravity anomaly")
ft.vis.axis('scaled')
ft.vis.contourf(xp, yp, gz, shape, 15)
ft.vis.colorbar(shrink=0.7)
ft.vis.m2km()

ft.vis.figure(figsize=(14,10))
ft.vis.subplots_adjust(top=0.95, left=0.05, right=0.95)
ft.vis.subplot(2, 3, 1)
ft.vis.title("x deriv (contour) + true (color map)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(xp, yp, gxz_true, shape, 12)
ft.vis.colorbar(shrink=0.7)
ft.vis.contour(xp, yp, gxz, shape, 12, color='k')
ft.vis.m2km()
ft.vis.subplot(2, 3, 2)
ft.vis.title("y deriv (contour) + true (color map)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(xp, yp, gyz_true, shape, 12)
ft.vis.colorbar(shrink=0.7)
ft.vis.contour(xp, yp, gyz, shape, 12, color='k')
ft.vis.m2km()
ft.vis.subplot(2, 3, 3)
ft.vis.title("z deriv (contour) + true (color map)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(xp, yp, gzz_true, shape, 8)
ft.vis.colorbar(shrink=0.7)
ft.vis.contour(xp, yp, gzz, shape, levels, color='k')
ft.vis.m2km()
ft.vis.subplot(2, 3, 4)
ft.vis.title("Difference x deriv")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, (gxz_true - gxz), shape)
ft.vis.colorbar(shrink=0.7)
ft.vis.m2km()
ft.vis.subplot(2, 3, 5)
ft.vis.title("Difference y deriv")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, (gyz_true - gyz), shape)
ft.vis.colorbar(shrink=0.7)
ft.vis.m2km()
ft.vis.subplot(2, 3, 6)
ft.vis.title("Difference z deriv")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, (gzz_true - gzz), shape)
ft.vis.colorbar(shrink=0.7)
ft.vis.m2km()
ft.vis.show()
