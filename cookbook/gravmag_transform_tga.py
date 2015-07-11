"""
GravMag: Calculate the analytic signal of a total field anomaly using FFT
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl

model = [mesher.Prism(-100, 100, -100, 100, 0, 2000, {'magnetization': 10})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)
z0 = -500
x, y, z = gridder.regular(area, shape, z=z0)
inc, dec = -30, 0
tf = utils.contaminate(prism.tf(x, y, z, model, inc, dec), 0.001,
                       percent=True)

# Need to convert gz to SI units so that the result is also in SI
total_grad_amp = transform.tga(x, y, utils.nt2si(tf), shape)

mpl.figure()
mpl.subplot(1, 2, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(y, x, tf, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.title("Total Gradient Amplitude")
mpl.axis('scaled')
mpl.contourf(y, x, total_grad_amp, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT/m')
mpl.m2km()
mpl.show()
