from __future__ import division
import numpy as np
from fatiando.gravmag import EulerDeconv, EulerDeconvEW, EulerDeconvMW
from fatiando.gravmag import sphere
from fatiando.mesher import Sphere
from fatiando import utils, gridder

model = None
xp, yp, zp = None, None, None
inc, dec = None, None
struct_ind = None
base = None
pos = None
field, dx, dy, dz = None, None, None, None
precision = 0.01


def setup():
    global model, x, y, z, inc, dec, struct_ind, field, dx, dy, dz, base, pos
    inc, dec = -30, 50
    pos = np.array([1000, 1200, 200])
    model = Sphere(pos[0], pos[1], pos[2], 1,
                   {'magnetization': utils.ang2vec(10000, inc, dec)})
    struct_ind = 3
    shape = (200, 200)
    x, y, z = gridder.regular((0, 3000, 0, 3000), shape, z=-100)
    base = 10
    field = sphere.tf(x, y, z, [model], inc, dec) + base
    # Use finite difference derivatives so that these tests don't depend on the
    # performance of the FFT derivatives.
    dx = (sphere.tf(x + 1, y, z, [model], inc, dec)
          - sphere.tf(x - 1, y, z, [model], inc, dec))/2
    dy = (sphere.tf(x, y + 1, z, [model], inc, dec)
          - sphere.tf(x, y - 1, z, [model], inc, dec))/2
    dz = (sphere.tf(x, y, z + 1, [model], inc, dec)
          - sphere.tf(x, y, z - 1, [model], inc, dec))/2


def test_euler_sphere_mag():
    "gravmag.EulerDeconv estimates center for sphere model and magnetic data"
    euler = EulerDeconv(x, y, z, field, dx, dy, dz, struct_ind).fit()
    assert (base - euler.baselevel_) / base <= precision, \
        'baselevel: %g estimated: %g' % (base, euler.baselevel_)
    assert np.all((pos - euler.estimate_) / pos <= precision), \
        'position: %s estimated: %s' % (str(pos), str(euler.estimate_))
    # Check if the R^2 metric (how good the fit is) is reasonably high
    # (best score is 1)
    data = -x*dx - y*dy - z*dz - struct_ind*field
    pred = euler.predicted()
    u = ((data - pred)**2).sum()
    v = ((data - data.mean())**2).sum()
    R2 = 1 - u/v
    assert R2 >= 0.999, "R^2 too low: {}".format(R2)


def test_euler_expandingwindow_sphere_mag():
    "gravmag.EulerDeconvEW estimates center for sphere model and magnetic data"
    euler = EulerDeconvEW(x, y, z, field, dx, dy, dz, struct_ind,
                          center=[1000, 1000],
                          sizes=np.linspace(100, 2000, 20))
    euler.fit()
    assert (base - euler.baselevel_) / base <= precision, \
        'baselevel: %g estimated: %g' % (base, euler.baselevel_)
    assert np.all((pos - euler.estimate_) / pos <= precision), \
        'position: %s estimated: %s' % (str(pos), str(euler.estimate_))


def test_euler_movingwindow_sphere_mag():
    "gravmag.EulerDeconvMW estimates center for sphere model and magnetic data"
    euler = EulerDeconvMW(x, y, z, field, dx, dy, dz, struct_ind,
                          windows=[10, 10], size=(1000, 1000), keep=0.2)
    euler.fit()
    for b in euler.baselevel_:
        assert (base - b) / base <= precision, \
            'baselevel: %g estimated: %g' % (base, b)
    for c in euler.estimate_:
        assert np.all((pos - c) / pos <= precision), \
            'position: %s estimated: %s' % (str(pos), str(c))
