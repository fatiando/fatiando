"""
Gridding: Extract a profile from map data
"""
from fatiando import gridder, utils
from fatiando.vis import mpl

# Generate random points
x, y = gridder.scatter((-2, 2, -2, 2), n=300, seed=1)
# And calculate 2D Gaussians on these points as sample data


def data(x, y):
    return (utils.gaussian2d(x, y, -0.6, -1)
            - utils.gaussian2d(x, y, 1.5, 1.5))
d = data(x, y)

# Extract a profile along the diagonal
p1, p2 = [-1.5, 0], [1.5, 1.5]
xp, yp, distance, dp = gridder.profile(x, y, d, p1, p2, 100)
dp_true = data(xp, yp)

mpl.figure()
mpl.subplot(2, 1, 2)
mpl.title("Irregular grid")
mpl.plot(xp, yp, '-k', label='Profile', linewidth=2)
mpl.contourf(x, y, d, (100, 100), 50, interp=True)
mpl.colorbar(orientation='horizontal')
mpl.legend(loc='lower right')
mpl.subplot(2, 1, 1)
mpl.title('Profile')
mpl.plot(distance, dp, '.b', label='Extracted')
mpl.plot(distance, dp_true, '-k', label='True')
mpl.xlim(distance.min(), distance.max())
mpl.legend(loc='lower right')
mpl.show()
