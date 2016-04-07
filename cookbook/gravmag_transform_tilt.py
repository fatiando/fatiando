"""
GravMag: Calculate the tilt angle of a total field anomaly
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl

model = [mesher.Prism(-200, 200, -200, 200, 0, 2000, {'magnetization': 10})]
area = (-5000, 5000, -5000, 5000)
shape = (200, 200)
z0 = -500
x, y, z = gridder.regular(area, shape, z=z0)
inc, dec = -30, 0
tf = prism.tf(x, y, z, model, inc, dec)
pole = transform.reduce_to_pole(x, y, tf, shape, inc, dec, sinc=inc, sdec=dec)
# Need to convert gz to SI units so that the result is also in SI
tilt_angle = transform.tilt(x, y, pole, shape)

mpl.figure()
mpl.subplot(1, 2, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(y, x, tf, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.title("Tilt Angle of Potential Field")
mpl.axis('scaled')
mpl.contourf(y, x, tilt_angle, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT/m')
mpl.m2km()
mpl.show()
