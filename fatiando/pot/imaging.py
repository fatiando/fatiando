"""
Imaging methods for potential fields.

Implements some of the methods described in Fedi and Pilkington (2012).

* :func:`~fatiando.pot.imaging.geninv`: The Generalized Inverse solver in the
  frequency domain for gravity data


**References**

Fedi, M., and M. Pilkington (2012), Understanding imaging methods for potential
field data, Geophysics, 77(1), G13, doi:10.1190/geo2011-0078.1

"""
import numpy

from fatiando.pot.fourier import _getfreqs
from fatiando.msh.ddd import PrismMesh
from fatiando.constants import G


def geninv(x, y, z, data, shape, zmin, zmax, nlayers):
    """
    Calculate a density distribution given gravity anomaly data on a **regular
    grid** using the Generalized Inverse solver in the frequency domain.

    .. note:: The data **must** be leveled, i.e., on the same height!

    ..note:: The coordinate system adopted is x->North, y->East, and z->Down

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * z : float
        The z coordinate of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid
    * zmin, zmax : float
        The top and bottom, respectively, of the region where the density
        distribution is calculated
    * nlayers : int
        The number of layers used to divide the region where the density
        distribution is calculated

    Returns:

    * mesh : :class:`fatiando.msh.ddd.PrismMesh`
        The estimated density distribution in a prism mesh (for easy 3D
        plotting). The estimated density is stored in ``mesh.props['density']``

    """
    if not isinstance(z, float):
        z = z[0]
    # Get the wavenumbers and the data Fourier transform
    Fx, Fy = _getfreqs(x, y, data, shape)
    freq = numpy.sqrt(Fx**2 + Fy**2)
    dataft = (2.*numpy.pi)*numpy.fft.fft2(numpy.reshape(data, shape))
    # Make a mesh fill with densities
    ny, nx = shape
    bounds = [x.min(), x.max(), y.min(), y.max(), zmin, zmax]
    mesh = PrismMesh(bounds, (nlayers, ny, nx))
    # Find the depth of the layers (middle of the prisms)
    zs = mesh.get_zs()
    dz = zs[1] - zs[0]
    depths = zs + 0.5*dz # Offset by the data height
    density = []
    for depth in depths:
        density.extend(
            numpy.real(
                numpy.fft.ifft2(
                    numpy.exp(-freq*depth)*freq*dataft/(numpy.pi*G)
                ).ravel()
            ))
    mesh.addprop('density', numpy.array(density))
    return mesh

