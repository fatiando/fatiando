import numpy as np

from fatiando.mesher import Sphere
from fatiando.gravmag import sphere
from fatiando import utils, gridder

model = None
xp, yp, zp = None, None, None
inc, dec = None, None
precision = 10 ** (-15)
lower_precision = 10 ** (-12)


def setup():
    global model, xp, yp, zp, inc, dec
    inc, dec = -30, 50
    reg_field = np.array(utils.dircos(inc, dec))
    model = [
        Sphere(500, 0, 1000, 1000,
               {'density': -1., 'magnetization': utils.ang2vec(-2, inc, dec)}),
        Sphere(-1000, 0, 700, 700,
               {'density': 2., 'magnetization': utils.ang2vec(5, 25, -10)})]
    xp, yp, zp = gridder.regular([-2000, 2000, -2000, 2000], (50, 50), z=-1)

# Need to come up with better test than cython vs numpy
# Ideas:
#   Laplace equation must be obeyed
#   Tensor and mag integrate to zero
#   potential should be the same all around
#   G vector should have same amplitude all around (this is good for all others #   as well)
