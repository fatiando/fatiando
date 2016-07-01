"""
GravMag: Calculate the tilt angle of a total field anomaly
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl
from fatiando.mesher import Prism


inc, dec = -90, 0
mag = utils.ang2vec(40, inc, dec)
model = [Prism(-1000, 1000, -200, 200, 0, 1000, {'magnetization': mag})]
area = [-5000, 5000, -5000, 5000]
shape = (200, 200)
x, y, z = gridder.regular(area, shape, z=-150)
data = prism.tf(x, y, z, model, inc, dec)
tilt_angle = transform.tilt(x, y, data, shape)

mpl.figure()
mpl.subplot(1, 2, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(y, x, data, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('nT')
mpl.m2km()
mpl.subplot(1, 2, 2)
mpl.title("Tilt")
mpl.axis('scaled')
mpl.contourf(y, x, tilt_angle, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar(orientation='horizontal').set_label('Radian')
mpl.m2km()
mpl.show()
