"""
GravMag: Calculating the derivatives of the gravity anomaly using FFT
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, fourier
from fatiando.vis import mpl

model = [mesher.Prism(-1000,1000,-1000,1000,0,2000,{'density':100})]
area = (-5000, 5000, -5000, 5000)
shape = (51, 51)
z0 = -500
xp, yp, zp = gridder.regular(area, shape, z=z0)
gz = utils.contaminate(prism.gz(xp, yp, zp, model), 0.001)

# Need to convert gz to SI units so that the result can be converted to Eotvos
gxz = utils.si2eotvos(fourier.derivx(xp, yp, utils.mgal2si(gz), shape))
gyz = utils.si2eotvos(fourier.derivy(xp, yp, utils.mgal2si(gz), shape))
gzz = utils.si2eotvos(fourier.derivz(xp, yp, utils.mgal2si(gz), shape))

gxz_true = prism.gxz(xp, yp, zp, model)
gyz_true = prism.gyz(xp, yp, zp, model)
gzz_true = prism.gzz(xp, yp, zp, model)

mpl.figure()
mpl.title("Original gravity anomaly")
mpl.axis('scaled')
mpl.contourf(xp, yp, gz, shape, 15)
mpl.colorbar(shrink=0.7)
mpl.m2km()

mpl.figure(figsize=(14,10))
mpl.subplots_adjust(top=0.95, left=0.05, right=0.95)
mpl.subplot(2, 3, 1)
mpl.title("x deriv (contour) + true (color map)")
mpl.axis('scaled')
levels = mpl.contourf(xp, yp, gxz_true, shape, 12)
mpl.colorbar(shrink=0.7)
mpl.contour(xp, yp, gxz, shape, 12, color='k')
mpl.m2km()
mpl.subplot(2, 3, 2)
mpl.title("y deriv (contour) + true (color map)")
mpl.axis('scaled')
levels = mpl.contourf(xp, yp, gyz_true, shape, 12)
mpl.colorbar(shrink=0.7)
mpl.contour(xp, yp, gyz, shape, 12, color='k')
mpl.m2km()
mpl.subplot(2, 3, 3)
mpl.title("z deriv (contour) + true (color map)")
mpl.axis('scaled')
levels = mpl.contourf(xp, yp, gzz_true, shape, 8)
mpl.colorbar(shrink=0.7)
mpl.contour(xp, yp, gzz, shape, levels, color='k')
mpl.m2km()
mpl.subplot(2, 3, 4)
mpl.title("Difference x deriv")
mpl.axis('scaled')
mpl.pcolor(xp, yp, (gxz_true - gxz), shape)
mpl.colorbar(shrink=0.7)
mpl.m2km()
mpl.subplot(2, 3, 5)
mpl.title("Difference y deriv")
mpl.axis('scaled')
mpl.pcolor(xp, yp, (gyz_true - gyz), shape)
mpl.colorbar(shrink=0.7)
mpl.m2km()
mpl.subplot(2, 3, 6)
mpl.title("Difference z deriv")
mpl.axis('scaled')
mpl.pcolor(xp, yp, (gzz_true - gzz), shape)
mpl.colorbar(shrink=0.7)
mpl.m2km()
mpl.show()
