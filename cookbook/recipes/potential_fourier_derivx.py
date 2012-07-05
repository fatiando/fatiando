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
#gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, prisms), 0.005)/SI2MGAL
gz = ft.pot.prism.gz(xp, yp, zp, prisms)/SI2MGAL

log.info("Calculating the x-derivative")
gxz = ft.pot.fourier.derivx(xp, yp, gz, shape)
gyz = ft.pot.fourier.derivy(xp, yp, gz, shape)
gzz = ft.pot.fourier.derivz(xp, yp, gz, shape)
print gzz, gzz.size
log.info("Computing true values of the derivative")
gxz_true = ft.pot.prism.gxz(xp, yp, zp, prisms)/SI2EOTVOS
gyz_true = ft.pot.prism.gyz(xp, yp, zp, prisms)/SI2EOTVOS
gzz_true = ft.pot.prism.gzz(xp, yp, zp, prisms)/SI2EOTVOS

log.info("Plotting")
ft.vis.figure()
ft.vis.subplot(3, 3, 2)
ft.vis.title("Original")
ft.vis.axis('scaled')
ft.vis.contourf(xp, yp, gz*SI2MGAL, shape, 15)
ft.vis.colorbar(shrink=0.9)
ft.vis.subplot(3, 3, 4)
ft.vis.title("Calculated (contour) + true (color map)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(xp, yp, gxz_true*SI2EOTVOS, shape, 12)
ft.vis.colorbar(shrink=0.9)
ft.vis.contour(xp, yp, gxz*SI2EOTVOS, shape, 12, color='k')
ft.vis.subplot(3, 3, 5)
ft.vis.title("Calculated (contour) + true (color map)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(xp, yp, gyz_true*SI2EOTVOS, shape, 12)
ft.vis.colorbar(shrink=0.9)
ft.vis.contour(xp, yp, gyz*SI2EOTVOS, shape, 12, color='k')
ft.vis.subplot(3, 3, 6)
ft.vis.title("Calculated (contour) + true (color map)")
ft.vis.axis('scaled')
levels = ft.vis.contourf(xp, yp, gzz_true*SI2EOTVOS, shape, 12)
ft.vis.colorbar(shrink=0.9)
ft.vis.contour(xp, yp, gzz*SI2EOTVOS, shape, levels, color='k')
ft.vis.subplot(3, 3, 7)
ft.vis.title("Calculated (contour) + true (color map)")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, (gxz_true - gxz)*SI2EOTVOS, shape)
ft.vis.colorbar(shrink=0.9)
ft.vis.subplot(3, 3, 8)
ft.vis.title("Calculated (contour) + true (color map)")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, (gyz_true - gyz)*SI2EOTVOS, shape)
ft.vis.colorbar(shrink=0.9)
ft.vis.subplot(3, 3, 9)
ft.vis.title("Calculated (contour) + true (color map)")
ft.vis.axis('scaled')
ft.vis.pcolor(xp, yp, (gzz_true - gzz)*SI2EOTVOS, shape)
ft.vis.colorbar(shrink=0.9)
ft.vis.show()
