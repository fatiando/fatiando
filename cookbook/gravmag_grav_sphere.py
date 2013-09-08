"""
GravMag: Forward modeling of the gravity anomaly and gravity gradient tensor
using spheres
"""
from fatiando import mesher, gridder, gravmag
from fatiando.vis import mpl

spheres = [mesher.Sphere(0, 0, 2000, 1000, {'density':1000})]
area = (-5000, 5000, -5000, 5000)
shape = (100, 100)
x, y, z = gridder.regular(area, shape, z=-100)
gz = gravmag.sphere.gz(x, y, z, spheres)
tensor = [gravmag.sphere.gxx(x, y, z, spheres),
          gravmag.sphere.gxy(x, y, z, spheres),
          gravmag.sphere.gxz(x, y, z, spheres),
          gravmag.sphere.gyy(x, y, z, spheres),
          gravmag.sphere.gyz(x, y, z, spheres),
          gravmag.sphere.gzz(x, y, z, spheres)]
mpl.figure()
mpl.axis('scaled')
mpl.title('gz')
mpl.contourf(y, x, gz, shape, 15)
mpl.colorbar()
mpl.m2km()
mpl.figure()
titles = ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
for i, field in enumerate(tensor):
    mpl.subplot(2, 3, i + 1)
    mpl.axis('scaled')
    mpl.title(titles[i])
    levels = mpl.contourf(y, x, field, shape, 15)
    mpl.colorbar()
    mpl.m2km()
mpl.show()
