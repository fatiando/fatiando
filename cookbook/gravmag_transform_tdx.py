"""
GravMag: Calculate the horizontal tilt angle of a potential field.
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl

model = [mesher.Prism(-100, 100, -100, 100, 0, 2000, {'magnetization': 10})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)
z0 = -500
xp, yp, zp = gridder.regular(area, shape, z=z0)
inc, dec = -30, 0
tf = utils.contaminate(prism.tf(xp, yp, zp, model, inc, dec), 0.001,
                       percent=True)

# Need to convert gz to SI units so that the result is also in SI
tilted_data = transform.tdx(xp, yp, utils.nt2si(tf), shape)
total_grad_amp = transform.tga(xp, yp, utils.nt2si(tf), shape)
tilted_tga = transform.thdr( xp, yp, total_grad_amp, shape )

mpl.figure()
mpl.subplot(2, 2, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(yp, xp, tf, shape, 30)
mpl.colorbar(orientation='horizontal')
mpl.m2km()
mpl.subplot(2, 2, 2)
mpl.title("TDX")
mpl.axis('scaled')
mpl.contourf(yp, xp, tilted_data, shape, 30)
mpl.colorbar(orientation='horizontal')
mpl.subplot(2, 2, 3)
mpl.title("Total Gradient Amplitude")
mpl.axis('scaled')
mpl.contourf(yp, xp, total_grad_amp, shape, 30)
mpl.colorbar(orientation='horizontal')
mpl.subplot(2, 2, 4)
mpl.title("TDX of TGA")
mpl.axis('scaled')
mpl.contourf(yp, xp, tilted_tga, shape, 30)
mpl.colorbar(orientation='horizontal')
mpl.m2km()
mpl.show()
