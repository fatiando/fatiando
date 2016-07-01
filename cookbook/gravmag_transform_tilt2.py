"""
GravMag: Calculate the tilt angle of a total field anomaly
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism, transform
from fatiando.vis import mpl

inc, dec = 45, 0
mag = utils.ang2vec(15, inc, dec)
model = [mesher.Prism(-3500, -2500, -1000, -600, 0, 500, {'magnetization': mag}),
    mesher.Prism(-500, 500, -200, 200, 300, 800, {'magnetization': mag}),
    mesher.Prism(2500, 3500, 600, 1000, 600, 1100, {'magnetization': mag})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)


#z0
z0 = -10
x, y, z = gridder.regular(area, shape, z=z0)
tf = prism.tf(x, y, z, model, inc, dec)
pole = transform.reduce_to_pole(x, y, tf, shape, inc, dec, sinc=inc, sdec=dec)
tilt_angle2 = transform.tilt(x, y, pole, shape)
#tilt_angle2 = transform.tilt(x, y, tf, shape)

#z-100
z0 = -100
x, y, z = gridder.regular(area, shape, z=z0)
tf = prism.tf(x, y, z, model, inc, dec)
pole = transform.reduce_to_pole(x, y, tf, shape, inc, dec, sinc=inc, sdec=dec)
tilt_angle3 = transform.tilt(x, y, pole, shape)
#tilt_angle3 = transform.tilt(x, y, tf, shape)

#z-500
z0 = -200
x, y, z = gridder.regular(area, shape, z=z0)
tf = prism.tf(x, y, z, model, inc, dec)
pole = transform.reduce_to_pole(x, y, tf, shape, inc, dec, sinc=inc, sdec=dec)
tilt_angle4 = transform.tilt(x, y, pole, shape)
#tilt_angle4 = transform.tilt(x, y, tf, shape)

mpl.figure()
mpl.subplot(2, 2, 1)
mpl.title("Original total field anomaly")
mpl.axis('scaled')
mpl.contourf(x, y, tf, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar().set_label('nT')
mpl.m2km()
mpl.subplot(2, 2, 2)
mpl.title("Tilt: z0 = -10")
mpl.axis('scaled')
mpl.contourf(x, y, tilt_angle2, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar().set_label('Radian')
mpl.contour(x, y, tilt_angle3, shape, levels=[0], clabel=False)
mpl.contour(x, y, tilt_angle2, shape, levels=[0.7853981633974482], clabel=False)
mpl.contour(x, y, tilt_angle2, shape, levels=[-0.7853981633974482], clabel=False)
mpl.m2km()
mpl.subplot(2, 2, 3)
mpl.title("Tilt: z0 = -100")
mpl.axis('scaled')
mpl.contourf(x, y, tilt_angle3, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar().set_label('Radian')
mpl.contour(x, y, tilt_angle3, shape, levels=[0], clabel=False)
mpl.contour(x, y, tilt_angle3, shape, levels=[0.7853981633974482], clabel=False)
mpl.contour(x, y, tilt_angle3, shape, levels=[-0.7853981633974482], clabel=False)
mpl.m2km()
mpl.subplot(2, 2, 4)
mpl.title("Tilt: z0 = -200")
mpl.axis('scaled')
mpl.contourf(x, y, tilt_angle4, shape, 30, cmap=mpl.cm.RdBu_r)
mpl.colorbar().set_label('Radian')
mpl.contour(x, y, tilt_angle3, shape, levels=[0], clabel=False)
mpl.contour(x, y, tilt_angle4, shape, levels=[0.7853981633974482], clabel=False)
mpl.contour(x, y, tilt_angle4, shape, levels=[-0.7853981633974482], clabel=False)
mpl.m2km()

mpl.show()
