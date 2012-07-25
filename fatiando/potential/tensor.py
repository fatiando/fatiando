r"""
Utilities for operating on the gradient tensor of potential fields.

**Eigen values, eigen vectors, and invariants**

* :func:`~fatiando.potential.tensor.invariants`: Calculates the first
  (:math:`I_1`), second (:math:`I_2`), and dimensionless (:math:`I`) invariants

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

Pedersen, L. B., and T. M. Rasmussen (1990), The gradient tensor of potential
field anomalies: Some implications on data collection and data processing of
maps, Geophysics, 55(12), 1558, doi:10.1190/1.1442807

----

"""
import numpy
import numpy.linalg


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

    * result : list = [eigval1, eigval2, eigval3, eigvec1, eigvec2, eigvec3]
        The eigenvalues and eigenvectors at each observation point.

        * eigval1,2,3 : array
            The first, second, and third eigenvalues
        * eigvec1,2,3 : array (shape = (N, 3) where N is the number of points)
            The first, second, and third eigenvectors

    Example::

        >>> tensor = [[2], [0], [0], [3], [0], [1]]
        >>> evl1, evl2, evl3, evc1, evc2, evc3 = eigen(tensor)
        >>> print evl1, evc1
        [ 3.] [[ 0.  1.  0.]]
        >>> print evl2, evc2
        [ 2.] [[ 1.  0.  0.]]
        >>> print evl3, evc3
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
    eigval1, eigval2, eigval3 = numpy.transpose(eigvals)
    eigvec1 = numpy.array(eigvec1)
    eigvec2 = numpy.array(eigvec2)
    eigvec3 = numpy.array(eigvec3)
    return eigval1, eigval2, eigval3, eigvec1, eigvec2, eigvec3

