"""
Load datasets from the internet.

**CRUST2.0**

Load and convert the `CRUST2.0 global crustal model
<http://igppweb.ucsd.edu/~gabi/rem.html>`_ (Bassin et al., 2000).

* :func:`~fatiando.datasets.fetch_crust2`: Download the .tar.gz archive with the model
  from the website
* :func:`~fatiando.datasets.crust2_to_tesseroids`: Convert the CRUST2.0 model to
  tesseroids

**Sample data**

Download a `Bouguer anomaly map of Alps (EGM 2008 model)
<https://gist.github.com/leouieda/6023922>`_ in Surfer ASCII grid file format.

* :func:`~fatiando.io.fetch_bouguer_alps_egm`: Download the .grd archive with
  the Bouguer anomaly of Alps (EGM 2008 model) from the website

**References**

Bassin, C., Laske, G. and Masters, G., The Current Limits of Resolution for
Surface Wave Tomography in North America, EOS Trans AGU, 81, F897, 2000.

----
"""
import urllib
import tarfile

import numpy

from . import gridder
from .mesher import Tesseroid


def fetch_crust2(fname='crust2.tar.gz'):
    """
    Download the CRUST2.0 model from http://igppweb.ucsd.edu/~gabi/crust2.html

    Parameters:

    * fname : str
        The name that the archive file will be saved when downloaded

    Returns:

    * fname : str
        The downloaded file name

    """
    urllib.urlretrieve('http://igpppublic.ucsd.edu/~gabi/ftp/crust2.tar.gz',
        filename=fname)
    return fname

def crust2_to_tesseroids(fname):
    """
    Convert the CRUST2.0 model to tesseroids.

    Opens the .tar.gz archive and converts the model to
    :class:`fatiando.mesher.Tesseroid`.
    Each tesseroid will have its ``props`` set to the apropriate Vp, Vs and
    density.

    The CRUST2.0 model includes 7 layers: ice, water, soft sediments, hard
    sediments, upper crust, middle curst and lower crust. It also includes the
    mantle below the Moho. The mantle portion is not included in this
    conversion because there is no way to place a bottom on it.

    Parameters:

    * fname : str
        Name of the model .tar.gz archive (see
        :func:`~fatiando.io.fetch_crust2`)

    Returns:

    * model : list of :class:`fatiando.mesher.Tesseroid`
        The converted model

    """
    archive = tarfile.open(fname, 'r:gz')
    # First get the topography and bathymetry information
    topogrd = _crust2_get_topo(archive)
    # Now make a dict with the codec for each type code
    codec = _crust2_get_codec(archive)
    # Get the type codes with the actual model
    types = _crust2_get_types(archive)
    # Convert to tesseroids
    size = 2
    lons = numpy.arange(-180, 180, size)
    lats = numpy.arange(90, -90, -size) # This is how lats are in the file
    model = []
    for i in xrange(len(lats)):
        for j in xrange(len(lons)):
            t = types[i][j]
            top = topogrd[i][j]
            for layer in xrange(7):
                if codec[t]['thickness'][layer] == 0:
                    continue
                w, e, s, n = lons[j], lons[j] + size, lats[i] - size, lats[i]
                bottom = top - codec[t]['thickness'][layer]
                props = {'density':codec[t]['density'][layer],
                         'vp':codec[t]['vp'][layer],
                         'vs':codec[t]['vs'][layer]}
                model.append(Tesseroid(w, e, s, n, top, bottom, props))
                top = bottom
    return model

def _crust2_get_topo(archive):
    """
    Fetch the matrix of topography and bathymetry from the CRUST2.0 archive.
    """
    f = archive.extractfile('./CNelevatio2.txt')
    topogrd = numpy.loadtxt(f, skiprows=1)[:, 1:]
    return topogrd

def _crust2_get_types(archive):
    """
    Fetch a matrix with the type code for each 2x2 degree cell.
    """
    f = archive.extractfile('./CNtype2.txt')
    typegrd = numpy.loadtxt(f, dtype=numpy.str, skiprows=1)[:, 1:]
    return typegrd

def _crust2_get_codec(archive):
    """
    Fetch the type code traslation codec from the archive and convert it to a
    dict.
    """
    f = archive.extractfile('./CNtype2_key.txt')
    # Skip the first 5 lines which are the header
    lines = [l.strip() for l in f.readlines()[5:] if l.strip()]
    # Each type code is 5 lines: code, vp, vs, density, thickness
    codec = {}
    for i in xrange(len(lines)/5):
        code = lines[i*5][:2]
        # Get the values and convert them to SI units
        vp = [float(v)*1000 for v in lines[i*5 + 1].split()]
        vs = [float(v)*1000 for v in lines[i*5 + 2].split()]
        density = [float(v)*1000 for v in lines[i*5 + 3].split()]
        # Skip the last thickness because it is an inf indicating the mantle
        thickness = [float(v)*1000 for v in lines[i*5 + 4].split()[:7]]
        codec[code] = {'vp':vp, 'vs':vs, 'density':density,
                       'thickness':thickness}
    return codec

def fetch_bouguer_alps_egm(fname='bouguer_alps_egm08.grd'):
    """
    Download the Bouguer anomaly of Alps (EGM 2008 model) in Surfer ASCII grid
    file format from https://gist.github.com/leouieda/6023922

    Parameters:

    * fname : str
        The name that the archive file will be saved when downloaded

    Returns:

    * fname : str
        The downloaded file name

    """
    urllib.urlretrieve('https://gist.github.com/leouieda/6023922/raw/' \
                       '948b0acbadb18e6ad49efe2092d9d9518b247780/' \
                       'bouguer_alps_egm08.grd', filename=fname)
    return fname
