"""
GravMag: Calculate the analytic signal of a total field anomaly using FFT
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl

model = [mesher.Prism(-3800, -3000, -3800, -3000, 0, -10, {'magnetization': 10}),
    mesher.Prism(-400, 400, -400, 400, 0, -30, {'magnetization': 10}),
    mesher.Prism(3800, 3000, 3800, 3000, 0, -60, {'magnetization': 10})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)
z0 = 0
x, y, z = gridder.regular(area, shape, z=z0)
inc, dec = -30, 0
tf = utils.contaminate(prism.tf(x, y, z, model, inc, dec), 0.001,
                       percent=True)
tf = prism.tf(x, y, z, model, inc, dec)
pole = transform.reduce_to_pole(x, y, tf, shape, inc, dec, sinc=inc, sdec=dec)
my_x = transform.derivx(x, y, z, shape, method='fft')
my_y = transform.derivy(x, y, z, shape, method='fft')

# Need to convert gz to SI units so that the result is also in SI
total_grad_amp = transform.tga(x, y, utils.nt2si(tf), shape)
tilt_angle = transform.tilt(x, y, pole, shape, xderiv=my_x, yderiv=my_y)

mpl.figure()
mpl.subplot(1, 3, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(y, x, tf, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT')
mpl.m2km()
mpl.subplot(1, 3, 2)
mpl.title("Total Gradient Amplitude")
mpl.axis('scaled')
mpl.contourf(y, x, total_grad_amp, shape, 30, cmap=mpl.cm.viridis)
mpl.colorbar(orientation='horizontal').set_label('nT/m')
mpl.m2km()
mpl.subplot(1, 3, 3)
mpl.title("Tilt")
mpl.axis('scaled')
mpl.contourf(y, x, tilt_angle, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('Radian')
mpl.m2km()
mpl.show()
