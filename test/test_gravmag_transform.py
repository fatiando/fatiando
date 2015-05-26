from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from fatiando.gravmag import transform, prism
from fatiando import gridder, utils
from fatiando.mesher import Prism


def _trim(array, shape, d=20):
    "Remove d elements from the edges of an array"
    return array.reshape(shape)[d : -d, d : -d].ravel()


def test_secont_horizontal_derivatives_fd():
    "gravmag.transform 2nd xy derivatives by finite diff against analytical"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-500)
    derivatives = 'xx yy'.split()
    grav = prism.potential(x, y, z, model)
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}'.format(deriv))(x, y, z, model)
        func = getattr(transform, 'deriv' + deriv[0])
        calculated = utils.si2eotvos(func(x, y, grav, shape, method='fd',
                                          order=2))
        diff = np.abs(analytical - calculated)
        assert np.all(diff/np.abs(analytical).max() <= 0.01), \
            "Failed for g{}. Max: {} Mean: {} STD: {}".format(
                deriv, diff.max(), diff.mean(), diff.std())


def test_horizontal_derivatives_fd():
    "gravmag.transform 1st xy derivatives by finite diff against analytical"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-200)
    derivatives = 'x y'.split()
    grav = utils.mgal2si(prism.gz(x, y, z, model))
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}z'.format(deriv))(x, y, z, model)
        func = getattr(transform, 'deriv' + deriv)
        calculated = utils.si2eotvos(func(x, y, grav, shape, method='fd'))
        diff = np.abs(analytical - calculated)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for g{}. Max: {} Mean: {} STD: {}".format(
                deriv, diff.max(), diff.mean(), diff.std())


def test_derivatives_uneven_shape():
    "gravmag.transform FFT derivatives work if grid spacing is uneven"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (150, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    grav = utils.mgal2si(prism.gz(x, y, z, model))
    analytical = prism.gzz(x, y, z, model)
    calculated = utils.si2eotvos(transform.derivz(x, y, grav, shape,
                                                  method='fft'))
    diff = _trim(np.abs(analytical - calculated), shape)
    assert np.all(diff <= 0.005*np.abs(analytical).max()), \
        "Failed for gzz"


def test_gz_derivatives():
    "gravmag.transform FFT 1st derivatives of gz against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'x y z'.split()
    grav = utils.mgal2si(prism.gz(x, y, z, model))
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}z'.format(deriv))(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv)(x, y, grav, shape,
                                                method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for g{}z".format(deriv)


def test_gx_derivatives():
    "gravmag.transform FFT 1st derivatives of gx against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'x y z'.split()
    grav = utils.mgal2si(prism.gx(x, y, z, model))
    for deriv in derivatives:
        analytical = getattr(prism, 'gx{}'.format(deriv))(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv)(x, y, grav, shape,
                                                method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for gx{}".format(deriv)


def test_gy_derivatives():
    "gravmag.transform FFT 1st derivatives of gy against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'x y z'.split()
    grav = utils.mgal2si(prism.gy(x, y, z, model))
    for deriv in derivatives:
        if deriv == 'x':
            func = getattr(prism, 'g{}y'.format(deriv))
        else:
            func = getattr(prism, 'gy{}'.format(deriv))
        analytical = func(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv)(x, y, grav, shape,
                                                method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for gy{}".format(deriv)


def test_second_derivatives():
    "gravmag.transform FFT second derivatives against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': -200})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'xx yy zz'.split()
    pot = prism.potential(x, y, z, model)
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}'.format(deriv))(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv[0])(x, y, pot, shape, order=2,
                                                   method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for g{}. Max: {} Mean: {} STD: {}".format(
                deriv, diff.max(), diff.mean(), diff.std())


def test_laplace_from_potential():
    "gravmag.transform 2nd derivatives of potential obey the Laplace equation"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 200})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    potential = prism.potential(x, y, z, model)
    gxx = utils.si2eotvos(transform.derivx(x, y, potential, shape, order=2,
                                           method='fft'))
    gyy = utils.si2eotvos(transform.derivy(x, y, potential, shape, order=2,
                                           method='fft'))
    gzz = utils.si2eotvos(transform.derivz(x, y, potential, shape, order=2))
    laplace = _trim(gxx + gyy + gzz, shape)
    assert np.all(np.abs(laplace) <= 1e-10), \
        "Max: {} Mean: {} STD: {}".format(
            laplace.max(), laplace.mean(), laplace.std())
