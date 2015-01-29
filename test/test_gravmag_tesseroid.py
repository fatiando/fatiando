from __future__ import division
import numpy as np
from numpy.testing import assert_almost_equal

from fatiando.gravmag import tesseroid, spherical_shell
from fatiando.mesher import Tesseroid, TesseroidMesh

shellmodel = None
halfshellmodel = None
halfshellmodel_eq = None
heights = None
density = None
props = None
top = None
bottom = None


def setup():
    "Make a spherical shell model with tesseroids"
    global shellmodel, halfshellmodel, halfshellmodel_eq, heights, density, \
        props, top, bottom
    heights = np.array([50e3, 250e3])
    density = 1000.
    props = {'density': density}
    top = 0
    bottom = -50000
    shellmodel = TesseroidMesh((0, 360, -90, 90, top, bottom), (1, 90, 180))
    shellmodel.addprop('density', density*np.ones(shellmodel.size))
    halfshellmodel = TesseroidMesh((0, 360, 0, 90, top, bottom), (1, 45, 180))
    halfshellmodel.addprop('density', density*np.ones(halfshellmodel.size))
    # Also make a half shell with axis at the equator
    halfshellmodel_eq = TesseroidMesh((-90, 90, -90, 90, top, bottom),
                                      (1, 90, 90))
    halfshellmodel_eq.addprop('density',
                              density*np.ones(halfshellmodel_eq.size))


def test_against_half_shell():
    "gravmag.tesseroid produces compatible results againt analytic half shell"
    fields = 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split()
    for f in fields:
        if f in 'gx gy'.split():
            shell = getattr(spherical_shell, 'half_gz')(heights, top, bottom,
                                                        density)
        elif f in 'gxy gxz gyz'.split():
            shell = getattr(spherical_shell, 'half_gzz')(heights, top, bottom,
                                                         density)
        else:
            shell = getattr(spherical_shell, 'half_' + f)(heights, top, bottom,
                                                          density)
        # Test things with the polar cap half shell
        lons = np.zeros_like(heights)
        lats = 90*np.ones_like(heights)
        tess = getattr(tesseroid, f)(lons, lats, heights, halfshellmodel)
        if f in 'gx gy gxy gxz gyz'.split():
            diff = 100*np.abs(tess)/np.abs(shell)
        else:
            diff = 100*np.abs(tess - shell)/np.abs(shell)
        for i in xrange(len(heights)):
            assert diff[i] <= 1, \
                "Failed {} at pole. ".format(f) + \
                "h={} shell={} tess={} diff={}".format(heights[i], shell[i],
                                                       tess[i], diff[i])
        # Test things with the equatorial cap half shell
        lons = np.zeros_like(heights)
        lats = np.zeros_like(heights)
        tess = getattr(tesseroid, f)(lons, lats, heights, halfshellmodel_eq)
        if f in 'gx gy gxy gxz gyz'.split():
            diff = 100*np.abs(tess)/np.abs(shell)
        else:
            diff = 100*np.abs(tess - shell)/np.abs(shell)
        for i in xrange(len(heights)):
            assert diff[i] <= 1, \
                "Failed {} at equator. ".format(f) + \
                "h={} shell={} tess={} diff={}".format(heights[i], shell[i],
                                                       tess[i], diff[i])
