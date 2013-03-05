r"""
Utilities for operating on the gradient tensor of potential fields.

**Functions**

* :func:`~fatiando.gravmag.tensor.invariants`: Calculates the first
  (:math:`I_1`), second (:math:`I_2`), and dimensionless (:math:`I`) invariants
* :func:`~fatiando.gravmag.tensor.eigen`: Calculates the eigenvalues and
  eigenvectors of the an array of gradient tensor measurements
* :func:`~fatiando.gravmag.tensor.center_of_mass`: Estimate the center of
  mass of sources from the first eigenvector using the method of
  Beiki and Pedersen (2010)

**Theory**

Following Pedersen and Rasmussen (1990), the characteristic polynomail of the
gravity gradient tensor :math:`\mathbf{\Gamma}` is

.. math::

    \lambda^3 + I_1\lambda - I_2 = 0

where :math:`\lambda` is an eigen value and :math:`I_1` and :math:`I_2` are
the two invariants. The dimensionless invariant :math:`I` is

.. math::

    I = -\dfrac{(I_2/2)^2}{(I_1/3)^3}

The invariant :math:`I` indicates the dimensionality of the source.
:math:`I = 0` for 2 dimensional bodies and :math:`I = 1` for a monopole.

**References**

Beiki, M., and L. B. Pedersen (2010), Eigenvector analysis of gravity gradient
tensor to locate geologic bodies, Geophysics, 75(6), I37, doi:10.1190/1.3484098

Pedersen, L. B., and T. M. Rasmussen (1990), The gradient tensor of potential
field anomalies: Some implications on data collection and data processing of
maps, Geophysics, 55(12), 1558, doi:10.1190/1.1442807

----

"""
import numpy
import numpy.linalg

from fatiando import gridder


def invariants(tensor):
    """
    Calculates the first, second, and dimensionless invariants of the gradient
    tensor.

    .. note:: The coordinate system used is x->North, y->East, z->Down

    Parameters:

    * tensor : list
        A list of arrays with the 6 components of the gradient tensor measured
        on a set of points. The order of the list should be:
        [gxx, gxy, gxz, gyy, gyz, gzz]

    Returns:

    * invariants : list = [:math:`I_1`, :math:`I_2`, :math:`I`]
        The invariants calculated for each point

    """
    gxx, gxy, gxz, gyy, gyz, gzz = tensor
    gyyzz = gyy*gzz
    gyz_sqr = gyz**2
    inv1 = gxx*gyy + gyyzz + gxx*gzz - gxy**2 - gyz_sqr - gxz**2
    inv2 = (gxx*(gyyzz - gyz_sqr) + gxy*(gyz*gxz - gxy*gzz)
            + gxz*(gxy*gyz - gxz*gyy))
    inv = -((0.5*inv2)**2)/((inv1/3.)**3)
    return [inv1, inv2, inv]

def eigen(tensor):
    """
    Calculates the eigenvalues and eigenvectors of the gradient tensor.

    .. note:: The coordinate system used is x->North, y->East, z->Down

    Parameters:

    * tensor : list
        A list of arrays with the 6 components of the gradient tensor measured
        on a set of points. The order of the list should be:
        [gxx, gxy, gxz, gyy, gyz, gzz]

    Returns:

    * result : list = [[eigval1, eigval2, eigval3], [eigvec1, eigvec2, eigvec3]]
        The eigenvalues and eigenvectors at each observation point.

        * eigval1,2,3 : array
            The first, second, and third eigenvalues
        * eigvec1,2,3 : array (shape = (N, 3) where N is the number of points)
            The first, second, and third eigenvectors

    Example::

        >>> tensor = [[2], [0], [0], [3], [0], [1]]
        >>> eigenvals, eigenvecs = eigen(tensor)
        >>> print eigenvals[0], eigenvecs[0]
        [ 3.] [[ 0.  1.  0.]]
        >>> print eigenvals[1], eigenvecs[1]
        [ 2.] [[ 1.  0.  0.]]
        >>> print eigenvals[2], eigenvecs[2]
        [ 1.] [[ 0.  0.  1.]]

    """
    eigvals = []
    eigvec1 = []
    eigvec2 = []
    eigvec3 = []
    for gxx, gxy, gxz, gyy, gyz, gzz in numpy.transpose(tensor):
        matrix = numpy.array([[gxx, gxy, gxz],
                              [gxy, gyy, gyz],
                              [gxz, gyz, gzz]])
        eigval, eigvec = numpy.linalg.eig(matrix)
        args = numpy.argsort(eigval)[::-1]
        eigvals.append([eigval[i] for i in args])
        eigvec1.append(eigvec[:,args[0]])
        eigvec2.append(eigvec[:,args[1]])
        eigvec3.append(eigvec[:,args[2]])
    eigvec1 = numpy.array(eigvec1)
    eigvec2 = numpy.array(eigvec2)
    eigvec3 = numpy.array(eigvec3)
    return numpy.transpose(eigvals), [eigvec1, eigvec2, eigvec3]

def center_of_mass(x, y, z, eigvec1, windows=1, wcenter=None, wmin=None,
    wmax=None):
    """
    Estimates the center of mass of a source using the method of Beiki and
    Pedersen (2010).

    Uses an expanding window to get the best estimate and deal with multiple
    sources.

    Parameters:

    * x, y, z : arrays
        The x, y, and z coordinates of the observation points
    * eigvec1 : array (shape = (N, 3) where N is the number of observations)
        The first eigenvector of the gravity gradient tensor at each observation
        point
    * windows : int
        The number of expanding windows to use
    * wcenter : list = [x, y]
        The [x, y] coordinates of the center of the expanding windows. Will
        default to the middle of the data area if None
    * wmin, wmax : float
        Minimum and maximum size of the expanding windows. Will default to
        10% data area and 100% data area, respectively, if None

    Returns:

    * [xo, yo, zo], sigma : float
        xo, yo, zo are the coordinates of the estimated center of mass. sigma is
        the estimated standard deviation of the distances between the estimated
        center of mass and the lines passing through the observation point in
        the direction of the eigenvector

    Example::

        >>> import fatiando as ft
        >>> # Generate synthetic data using a prism
        >>> prism = ft.mesher.Prism(-200,0,-100,100,0,200,{'density':1000})
        >>> x, y, z = ft.gridder.regular((-500,500,-500,500), (20,20), z=-100)
        >>> tensor = [ft.gravmag.prism.gxx(x, y, z, [prism]),
        ...           ft.gravmag.prism.gxy(x, y, z, [prism]),
        ...           ft.gravmag.prism.gxz(x, y, z, [prism]),
        ...           ft.gravmag.prism.gyy(x, y, z, [prism]),
        ...           ft.gravmag.prism.gyz(x, y, z, [prism]),
        ...           ft.gravmag.prism.gzz(x, y, z, [prism])]
        >>> # Get the eigenvector
        >>> eigenvals, eigenvecs = ft.gravmag.tensor.eigen(tensor)
        >>> # Now estimate the center of mass
        >>> cm, sigma = ft.gravmag.tensor.center_of_mass(x, y, z, eigenvecs[0])
        >>> xo, yo, zo = cm
        >>> print "%.2lf, %.2lf, %.2lf" % (xo, yo, zo)
        -100.05, 0.00, 99.86

    """
    if wmin is None:
        wmin = 0.1*numpy.mean([x.max() - x.min(), y.max() - y.min()])
    if wmax is None:
        wmax = numpy.mean([x.max() - x.min(), y.max() - y.min()])
    if wcenter is None:
        wcenter = [0.5*(x.min() + x.max()), 0.5*(y.min() + y.max())]
    xc, yc = wcenter
    best = None
    for size in numpy.linspace(wmin, wmax, windows):
        area = [xc - 0.5*size, xc + 0.5*size, yc - 0.5*size, yc + 0.5*size]
        wx, wy, scalars = gridder.cut(x, y, [z, eigvec1], area)
        wz, weigvec1 = scalars
        # Estimate the center of mass for the data in this window
        vx, vy, vz = numpy.transpose(weigvec1)
        m11 = numpy.sum(1 - vx**2)
        m12 = numpy.sum(-vx*vy)
        m13 = numpy.sum(-vx*vz)
        m22 = numpy.sum(1 - vy**2)
        m23 = numpy.sum(-vy*vz)
        m33 = numpy.sum(1 - vz**2)
        matrix = numpy.array(
            [[m11, m12, m13],
             [m12, m22, m23],
             [m13, m23, m33]])
        vector = numpy.array([
            numpy.sum((1 - vx**2)*wx - vx*vy*wy - vx*vz*wz),
            numpy.sum(-vx*vy*wx + (1 - vy**2)*wy - vy*vz*wz),
            numpy.sum(-vx*vz*wx - vy*vz*wy + (1 - vz**2)*wz)])
        cm = numpy.linalg.solve(matrix, vector)
        xo, yo, zo = cm
        dists = ((xo - wx)**2 + (yo - wy)**2 + (zo - wz)**2 -
                 ((xo - wx)*vx + (yo - wy)*vy + (zo - wz)*vz)**2)
        sigma = numpy.sqrt(numpy.sum(dists)/len(wx))
        if best is None or sigma < best[1]:
            best = [cm, sigma]
    return best
