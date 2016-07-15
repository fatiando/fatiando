import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises

from fatiando.seismic import conv


def test_impulse_response():
    """
    conv.convolutional_model raises the source wavelet as result when the model
    is a centred spike, considering the dimension of the model equal to the
    source wavelet
    """
    w = conv.rickerwave(30., 2.e-3)
    rc_test = np.zeros((w.shape[0], 20))
    rc_test[w.shape[0]/2, :] = 1.
    spike = conv.convolutional_model(rc_test, 30., conv.rickerwave, dt=2.e-3)
    for j in range(0, rc_test.shape[1]):
        assert_array_almost_equal(spike[:, j], w, 9)


def test_rc_shorter_than_wavelet():
    """
    When the reflectivity series is shorter than the wavelength, the spike
    response is observed like in the opposite case. The difference is that the
    the ricker wavelet (or other symmetric wavelet) is shorter in the result.
    """
    w = conv.rickerwave(30., 2.e-3)
    rc_test = np.zeros((21, 20))
    rc_test[rc_test.shape[0]/2, :] = 1
    spike = conv.convolutional_model(rc_test, 30., conv.rickerwave, dt=2.e-3)
    for j in range(0, rc_test.shape[1]):
        assert_array_almost_equal(spike[:, j],
                                  w[(w.shape[0]-rc_test.shape[0])/2:
                                  -(w.shape[0]-rc_test.shape[0])/2], 9)


def test_reflectivity_wrong_dimensions():
    """
    Velocity and density are provided as matrix or vector to reflectivity
    calculation, so they must have the same dimension.
    """
    vel = np.ones((10, 10))
    dens = np.ones((11, 11))
    raises(AssertionError, conv.reflectivity, vel, dens)
    vel = np.ones((10))
    dens = np.ones((11))
    raises(AssertionError, conv.reflectivity, vel, dens)


def test_depth_2_time_wrong_dimensions():
    """
    Velocity and property are provided as matrix to depth to time cconversion,
    so they must have the same dimension.
    """
    vel = np.ones((10, 10))
    dens = np.ones((11, 11))
    dt = 2.e-3
    dz = 1.
    raises(AssertionError, conv.depth_2_time, vel, dens, dt, dz)


def test_ricker():
    """
    conv.rickerwave inputs must satisfy the condition for sampling and
    stability, otherwise this implies in a error.
    """
    f = 50.
    dt = 2.e-3
    raises(AssertionError, conv.rickerwave, f, dt)
