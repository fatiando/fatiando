import numpy as np

from fatiando import gravmag
from fatiando.mesher import Tesseroid
#from fatiando.gravmag import _tesseroid, _ctesseroid

shellmodel = None
heights = None
density = None
props = None
top = None
bottom = None

def setup():
    "Make a spherical shell model with tesseroids"
    global shellmodel, heights, density, props, top, bottom
    tlons = np.linspace(-90, 90, 50, endpoint=False)
    tlats = np.linspace(-90, 90, 50, endpoint=False)
    wsize = tlons[1] - tlons[0]
    ssize = tlats[1] - tlats[0]
    density = 1000.
    props = {'density':density}
    top = 0
    bottom = -50000
    shellmodel = [Tesseroid(w, w + wsize, s, s + ssize, top, bottom, props)
        for w in tlons for s in tlats]
    heights = np.linspace(250000, 1000000, 10)

def test_potential():
    "gravmag.tesseroid.potential with optimal discretize against half a shell"
    shell = gravmag.half_sph_shell.potential(heights, top, bottom, density)
    lons = np.zeros_like(heights)
    lats = lons
    tess = gravmag.tesseroid.potential(lons, lats, heights, shellmodel)
    diff = np.abs((shell - tess)/shell)
    assert np.all(diff <= 0.01), 'diff: %s' % (str(diff))

def test_gz():
    "gravmag.tesseroid.gz with optimal discretize against half a shell"
    shell = gravmag.half_sph_shell.gz(heights, top, bottom, density)
    lons = np.zeros_like(heights)
    lats = lons
    tess = gravmag.tesseroid.gz(lons, lats, heights, shellmodel)
    diff = np.abs(shell - tess)/np.abs(shell)
    assert np.all(diff <= 0.01), 'diff: %s' % (str(diff))


def test_gzz():
    "gravmag.tesseroid.gzz with optimal discretize against half a shell"
    shell = gravmag.half_sph_shell.gzz(heights, top, bottom, density)
    lons = np.zeros_like(heights)
    lats = lons
    tess = gravmag.tesseroid.gzz(lons, lats, heights, shellmodel)
    diff = np.abs(shell - tess)/np.abs(shell)
    assert np.all(diff <= 0.01), 'diff: %s' % (str(diff))
