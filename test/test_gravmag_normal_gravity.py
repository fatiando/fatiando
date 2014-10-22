from __future__ import division
from fatiando.gravmag.normal_gravity import (WGS84,
                                             gamma_somigliana,
                                             gamma_somigliana_free_air,
                                             gamma_closed_form)

from fatiando import utils

import numpy
from numpy.testing import assert_almost_equal


def test_somigliana():
    "gravmag.normal_gravity.gamma_somigliana computes consistent results"
    res = gamma_somigliana(0, ellipsoid=WGS84)
    assert res == utils.si2mgal(WGS84.gamma_a), \
        "somigliana at equator != from gamma_a: {:.20f}".format(res)
    res = gamma_somigliana(90, ellipsoid=WGS84)
    assert res == utils.si2mgal(WGS84.gamma_b), \
        "somigliana at north pole != from gamma_b: {:.20f}".format(res)
    res = gamma_somigliana(-90, ellipsoid=WGS84)
    assert res == utils.si2mgal(WGS84.gamma_b), \
        "somigliana at south pole != from gamma_b: {:.20f}".format(res)


def test_free_air():
    "gravmag.normal_gravity.gamma_somigliana_free_air compatible w closed form"
    for h in [0, 10, 100]:
        fa = gamma_somigliana_free_air(45, h, ellipsoid=WGS84)
        closed = gamma_closed_form(45, h, ellipsoid=WGS84)
        assert_almost_equal(fa, closed, err_msg='at {} m'.format(h), decimal=1)


def test_closed_form():
    "gravmag.normal_gravity.gamma_closed_form compatible with somigliana"
    lat = numpy.linspace(-90, 90, 200)
    som = gamma_somigliana(lat, ellipsoid=WGS84)
    closed = gamma_closed_form(lat, 0, ellipsoid=WGS84)
    for i in xrange(len(lat)):
        assert_almost_equal(closed[i], som[i], decimal=3,
                            err_msg='lat = {}'.format(lat[i]))

    gradient = (gamma_closed_form(lat, 1, ellipsoid=WGS84)
                - gamma_closed_form(lat, 0, ellipsoid=WGS84))
    mean = numpy.mean(gradient)
    assert_almost_equal(mean, -0.3086, decimal=4, err_msg='mean vs free-air')

    gamma_a = gamma_closed_form(0, 0, ellipsoid=WGS84)
    assert_almost_equal(gamma_a, utils.si2mgal(WGS84.gamma_a), decimal=5,
                        err_msg='equator vs gamma_a')

    gamma_b = gamma_closed_form(90, 0, ellipsoid=WGS84)
    assert_almost_equal(gamma_b, utils.si2mgal(WGS84.gamma_b), decimal=5,
                        err_msg='north pole vs gamma_b')

    gamma_b = gamma_closed_form(-90, 0, ellipsoid=WGS84)
    assert_almost_equal(gamma_b, utils.si2mgal(WGS84.gamma_b), decimal=5,
                        err_msg='south pole vs gamma_b')
