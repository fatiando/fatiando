"""
Input/Output utilities for grids, models, etc

**CRUST2.0**

Load and convert the `CRUST2.0 global crustal model 
<http://igppweb.ucsd.edu/~gabi/rem.html>`_ (Bassin et al., 2000).

* :func:`~fatiando.io.fetch_crust2`: Download the .tar.gz archive with the model
  from the website
* :func:`~fatiando.io.crust2_to_tesseroids`: Convert the CRUST2.0 model to 
  tesseroids 

**References**

Bassin, C., Laske, G. and Masters, G., The Current Limits of Resolution for 
Surface Wave Tomography in North America, EOS Trans AGU, 81, F897, 2000.

----
"""
import urllib
import tarfile

import numpy

from fatiando.mesher import Tesseroid


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
        fname=fname)
    return fname

def curst2_to_tesseroids(fname):
    """
    Convert the CRUST2.0 model to tesseroids.

    Opens the .tar.gz archive and converts the model to 
    :class:`fatiando.mesher.Tesseroid`. 
    Each tesseroid will have its ``props`` set to the apropriate Vp, Vs and 
    density.

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
    lons = numpy.arange(-180, 180, 2)
    lats = numpy.arange(-90, 90, 2)
    model = []
    for i in xrange(len(lats)):
        for j in xrange(len(lons)):
            t = types[i][j]
            for layer in xrange(8):
                model.append(
                    Tesseroid(   
 
def _crust2_get_topo(archive):
    """
    Fetch the matrix of topography and bathymetry from the CRUST2.0 archive.
    """
    f = archive.extractfile('CNevelvatio2.txt')
    
