"""
GravMag: Generate synthetic gravity data on an irregular grid
"""
from fatiando import mesher, gridder
from fatiando.vis import mpl
from fatiando.gravmag import prism

model = [mesher.Prism(-2000, 2000, -2000, 2000, 0, 2000, {'density':1000})]
xp, yp, zp = gridder.scatter((-5000, 5000, -5000, 5000), n=100, z=-100)
gz = prism.gz(xp, yp, zp, model)

shape = (100,100)
mpl.axis('scaled')
mpl.title("gz produced by prism model on an irregular grid (mGal)")
mpl.plot(xp, yp, '.k', label='Grid points')
levels = mpl.contourf(xp, yp, gz, shape, 12, interp=True)
mpl.contour(xp, yp, gz, shape, levels, interp=True)
mpl.legend(loc='lower right', numpoints=1)
mpl.m2km()
mpl.show()
