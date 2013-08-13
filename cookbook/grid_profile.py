"""
Gridding: Extract a profile of the data using interpolation
"""
import numpy as np
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate random points
x, y = gridder.scatter((-2, 2, -2, 2), n=300, seed=1)
# And calculate 2D Gaussians on these points as sample data
def data(x, y):
    return (utils.gaussian2d(x, y, -0.6, -1)
            - utils.gaussian2d(x, y, 1.5, 1.5))
z = data(x, y)

# Extract a profile along the diagonal
xp = np.linspace(-2, 2, 100)
yp = xp
zp = gridder.interp_at(x, y, z, xp, yp)
zp_true = data(xp, yp)

mpl.figure()
mpl.subplot(2, 1, 2)
mpl.title("Irregular grid")
mpl.plot(x, y, '.k')
mpl.plot(xp, yp, '-k', label='Profile')
mpl.contourf(x, y, z, (100, 100) , 50, interp=True)
mpl.colorbar(orientation='horizontal')
mpl.legend(loc='lower right')
mpl.subplot(2, 1, 1)
mpl.title('Profile')
mpl.plot(xp, zp, '.b', label='Interpolated')
mpl.plot(xp, zp_true, '-k', label='True')
mpl.legend(loc='lower right')
mpl.show()
