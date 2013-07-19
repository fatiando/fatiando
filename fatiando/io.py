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
    
**Surfer Grid File**

Load an Surfer ASCII grid file.

* :func:`~fatiando.io.load_surfer`: Read a Surfer grid file and return three 1d 
  numpy arrays and the grid shape
        
**EGM2008**

Download the `Bouguer anomaly of Alps (EGM 2008 model)
<https://gist.github.com/leouieda/6023922>`_ in a Surfer ASCII grid format.
    
* :func:`~fatiando.io.fetch_bouguer_alps_egm`: Download the .grd archive with 
  the Bouguer anomaly of Alps (EGM 2008 model) from the website
    
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

def load_surfer(fname, fmt='ascii'):
    """
    Read a Surfer grid file and return three 1d numpy arrays and the grid shape.
      
    Surfer is a contouring, gridding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.
    
    http://www.goldensoftware.com/products/surfer
        
    According to Surfer structure, x and y are horizontal and vertical
    screen-based coordinates respectively. If the grid is in geographic
    coordinates, x will be longitude and y latitude. If the coordinates
    are cartesian, x will be the easting and y the norting coordinates.
        
    WARNING: This is opposite to the convention used for Fatiando. 
    See io_surfer.py in cookbook.
    
    Parameters:
    
    * fname : str
        Name of the Surfer grid file
    * fmt : str
        File type, can be 'ascii' or 'binary'
    
    Returns:
    
    * x : 1d-array
        Value of the horizontal coordinate of each grid point.
    * y : 1d-array
        Value of the vertical coordinate of each grid point.
    * grd : 1d-array
        Values of the field in each grid point. Field can be for example
        topography, gravity anomaly etc
    * shape : shape of the grid in the format (ny, nx)
        
    """
    assert fmt in ['ascii', 'binary'], "Invalid grid format '%s'. Should be \
        'ascii' or 'binary'." % (fmt)
    if fmt == 'ascii':
        # Surfer ASCII grid structure
        # DSAA            Surfer ASCII GRD ID
        # nCols nRows     number of columns and rows
        # xMin xMax       X min max
        # yMin yMax       Y min max
        # zMin zMax       Z min max
        # z11 z21 z31 ... List of Z values
        with open(fname) as ftext:
            # DSAA is a Surfer ASCII GRD ID
            id = ftext.readline()
            # Read the number of columns (nx) and rows (ny)
            nx, ny = [int(s) for s in ftext.readline().split()]
            # Read the min/max value of x (columns/longitue)
            xmin, xmax = [float(s) for s in ftext.readline().split()]
            # Read the min/max value of  y(rows/latitude)
            ymin, ymax = [float(s) for s in ftext.readline().split()]
            # Read the min/max value of grd
            zmin, zmax = [float(s) for s in ftext.readline().split()]
            data = numpy.fromiter((float(i) for line in ftext for i in
                                   line.split()), dtype='f')
            grd = numpy.ma.masked_greater_equal(data, 1.70141e+38)
        # Create x and y numpy arrays
        x = numpy.linspace(xmin, xmax, nx)
        y = numpy.linspace(ymin, ymax, ny)
        x, y = [tmp.ravel() for tmp in numpy.meshgrid(x, y)]
    if fmt == 'binary':
        raise NotImplementedError("Binary file support is not implemented yet. \
                                  Sorry")
    return x, y, grd, (ny,nx)

def fetch_bouguer_alps_egm(fname='bouguer_alps_egm08.grd'):
    """
    Download the Bouguer anomaly of Alps (EGM 2008 model) from
    from https://gist.github.com/leouieda/6023922
    
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