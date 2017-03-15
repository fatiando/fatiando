"""
Apply padding to data grids using different algorithms for the filling.
"""
from __future__ import division, absolute_import, print_function
from future.builtins import range
import numpy as np


def pad_array(a, npd=None, padtype='OddReflectionTaper'):
    """
    Return a padded array of arbitrary dimension.

    The function takes an array of arbitrary dimension and pads it either to
    the dimensions given by the tuple *npd*, or to the next power of 2 if *npd*
    is not given.

    An odd reflection with a cosine taper (``padtype='OddReflectionTaper'``) is
    the preferred method of padding for Fourier Transform operations. The odd
    reflection optimally preserves the frequency content while adding minimal
    sharp inflections.

    .. note::

        Requires gridded data of the same dimension as the desired output
        (i.e. no flattened arrays; use reshape).

    .. note:: This function returns a deep copy of the original array.

    Parameters:

    * a : array
        Array (N-D) to be padded
    * npd : tuple (optional)
        Desired shape of new padded array.  If not provided, the nearest
        power of 2 will be used.
    * padtype : string (optional)
        What method will be used to pad the new values. Can be lower or upper
        case. Options:

        * *oddreflectiontaper*: Generates odd reflection then tapers to the
          mean using a cosine function (Default)
        * *oddreflection*: Pads with the odd reflection, with no taper
        * *reflection*: Pads with simple reflection
        * *lintaper*: Linearly tapers to the mean
        * *value*: Numeric value (e.g., ``'10.4'``). Input a float or integer
          directly.
        * *edge*: Uses the edge value as a constant pad
        * *mean*: Uses the mean of the vector along each axis

    Returns:

    * ap : numpy array
        Padded array. The array core is a deep copy of the original array
    * nps : list
        List of tuples containing the number of elements padded onto each
        dimension.

    Examples:

    >>> import numpy as np
    >>> z = np.array([3, 4, 4, 5, 6])
    >>> zpad, nps = pad_array(z)
    >>> print(zpad)
    [ 4.4  3.2  3.   4.   4.   5.   6.   4.4]
    >>> print(nps)
    [(2, 1)]

    >>> shape = (5, 6)
    >>> z = np.ones(shape)
    >>> zpad, nps = pad_array(z, padtype='5')
    >>> zpad
    array([[ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
           [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.],
           [ 5.,  1.,  1.,  1.,  1.,  1.,  1.,  5.],
           [ 5.,  1.,  1.,  1.,  1.,  1.,  1.,  5.],
           [ 5.,  1.,  1.,  1.,  1.,  1.,  1.,  5.],
           [ 5.,  1.,  1.,  1.,  1.,  1.,  1.,  5.],
           [ 5.,  1.,  1.,  1.,  1.,  1.,  1.,  5.],
           [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.]])
    >>> print(nps)
    [(2, 1), (1, 1)]

    """
    # Test to make sure padtype is valid
    padopts = ['oddreflectiontaper', 'oddreflection', 'reflection',
               'lintaper', 'edge', 'value', 'mean']
    # API Note:
    # If one wishes to add more options to the padding, use the following
    # checklist to make sure all the sections are consistent.
    #  [ ] Docstring (2 spots: the options, and the option descriptions
    #  [ ] padopts - Make sure to add to avoid raising a ValueError
    #  [ ] if statements below - Finally add an elif to the if statements
    #                            below. Add a descriptive comment.
    if str(padtype).lower() not in padopts and not _is_number(padtype):
        raise ValueError('Invalid padtype "{}"'.format(padtype))
    # If npd is not provided, populate with next power of 2
    npt = []
    nd = a.ndim
    if npd is None:
        for ii in range(0, nd):
            if nd == 1:
                npt.append(_nextpow2(len(a)))
            else:
                npt.append(_nextpow2(a.shape[ii]))
    else:
        et = 'Pad dimensions do not match array dims'
        if nd == 1:
            if _is_integer(npd):
                npt.append(npd)
            else:
                if len(npd) != 1:
                    raise ValueError(et)
                else:
                    npt.extend(npd)
        else:
            if _is_integer(npd):
                raise ValueError(et)
            elif len(npd) != a.ndim:
                raise ValueError(et)
            else:
                npt = npd
        for ii in range(0, len(npt)):
            if npt[ii] <= a.shape[ii]:
                raise ValueError(
                    'Desired padding is less than array ' +
                    'length along dimension' + str(ii) + '.')
    # Compute numbers to pad on the left and right side of the array along
    # each dimension
    nps = []
    for ii in range(0, nd):
        nps.append((int(np.ceil((npt[ii] - a.shape[ii])/2.)),
                    int(np.floor((npt[ii] - a.shape[ii])/2.))))

    # If it will be needed, compute the mean
    meanneeded = ['lintaper', 'oddreflectiontaper']
    if padtype.lower() in meanneeded:
        m = np.mean(a)
    # Use numpy's padding routines where possible
    if _is_number(padtype):
        # Pad with value
        ap = np.pad(a, nps, mode='constant',
                    constant_values=(float(padtype), float(padtype)))
    elif padtype.lower() == 'mean':
        # Pad with the mean
        ap = np.pad(a, nps, mode='mean')
    elif padtype.lower() == 'lintaper':
        # Linearly taper to the mean
        ap = np.pad(a, nps, mode='linear_ramp', end_values=(m, m))
    elif padtype.lower() == 'edge':
        # Pad with edge values
        ap = np.pad(a, nps, mode='edge')
    elif padtype.lower() == 'reflection':
        # Pad with even reflection
        ap = np.pad(a, nps, mode='reflect', reflect_type='even')
    elif padtype.lower() == 'oddreflection':
        # Pad with odd reflection
        ap = np.pad(a, nps, mode='reflect', reflect_type='odd')
    elif padtype.lower() == 'oddreflectiontaper':
        # Pad with odd reflection and a cosine taper to mean
        ap = (np.pad(a, nps, mode='reflect', reflect_type='odd') - m)
        for ii in range(0, nd):
            ap = np.apply_along_axis(_costaper, ii, ap, lp=nps[ii][0],
                                     rp=nps[ii][1])
        ap += m

    return ap, nps


def unpad_array(a, nps):
    """
    Remove padding from an array.

    This function takes a padded array and removes the padding from both sides.
    Designed to use the output of :func:`~fatiando.gridder.pad_array`.

    .. note::

        Unlike :func:`~fatiando.gridder.pad_array`, this function **returns a
        slice** of the input array. Therefore, any changes to the padded array
        will be reflected in the unpadded array!

    Parameters:

    * a : array
        Array to be un-padded.  Can be of arbitrary dimension.
    * nps : list
        List of tuples giving the min and max indices for the cutoff.
        Use the value returned by :func:`~fatiando.gridder.pad_array`.

    Returns:

    * b : array
        Array of same dimension as a, with padding removed

    Examples:

    >>> import numpy as np
    >>> z = np.array([3, 4, 4, 5, 6])
    >>> zpad, nps = pad_array(z)
    >>> print(zpad)
    [ 4.4  3.2  3.   4.   4.   5.   6.   4.4]
    >>> zunpad = unpad_array(zpad, nps)
    >>> print(zunpad)
    [ 3.  4.  4.  5.  6.]

    """
    o = []
    for ii in range(0, a.ndim):
        o.append(slice(nps[ii][0], a.shape[ii] - nps[ii][1]))
    b = a[o]

    return b


def pad_coords(xy, shape, nps):
    """
    Apply padding to coordinate vectors.

    Designed to be used in concert with :func:`~fatiando.gridder.pad_array`,
    this function takes a list of coordinate vectors and pads them using the
    same number of elements as the padding of the data array.

    .. note::

        This function returns a list of arrays in the same format as, for
        example, :func:`~fatiando.gridder.regular`. It is a list of flattened
        ``np.meshgrid`` for each vector in the same order as was input through
        argument *xy*.

    Parameters:

    * xy : list
        List of arrays of coordinates
    * shape : tuple
        Size of original array
    * nps : list
        List of tuples containing the number of elements padded onto each
        dimension (use the output from :func:`~fatiando.gridder.pad_array`).

    Returns:

    * coordspad : list
        List of padded coordinate arrays

    Examples:

    >>> import numpy as np
    >>> from fatiando.gridder import regular
    >>> shape = (5, 6)
    >>> x, y, z = regular((-10, 10, -20, 0), shape, z=-25)
    >>> gz = np.zeros(shape)
    >>> gzpad, nps = pad_array(gz)
    >>> print(x.reshape(shape)[:, 0])
    [-10.  -5.   0.   5.  10.]
    >>> print(y.reshape(shape)[0, :])
    [-20. -16. -12.  -8.  -4.   0.]
    >>> xy = [x, y]
    >>> N = pad_coords(xy, shape, nps)
    >>> print(N[0].reshape(gzpad.shape)[:, 0])
    [-20. -15. -10.  -5.   0.   5.  10.  15.]
    >>> print(N[1].reshape(gzpad.shape)[0, :])
    [-24. -20. -16. -12.  -8.  -4.   0.   4.]

    """
    coords = []
    d = []
    coordspad = []
    for ii in range(0, len(shape)):
        if type(xy) is not list:
            coords.append(xy)
        elif type(xy) is list and len(shape) > 1:
            coords.append(xy[ii].reshape(shape).transpose().take(0, axis=ii))
        d.append(coords[ii][1] - coords[ii][0])
        coordspad.append(_padcvec(coords[ii], nps[ii], d[ii]))
    M = np.meshgrid(*[a for a in tuple(coordspad)[::-1]])
    N = []
    for a in M:
        N.append(a.ravel())
    return N[::-1]


def _padcvec(x, n, dx):
    # Used by padcoords to pad an individual vector based on the number
    # of points on either side and the point spacing
    xp = np.zeros(len(x) + n[0] + n[1])
    xp[n[0]:n[0]+len(x)] = x[:]
    for ii, jj in enumerate(list(range(0, n[0]))[::-1]):
        xp[ii] = x[0] - ((jj + 1) * dx)
    for ii, jj in enumerate(range(len(x) + n[0], len(xp))):
        xp[jj] = x[-1] + (dx * (ii + 1))
    return xp


def _costaper(a, lp, rp):
    # This takes an array and applies a cosine taper to each end.
    # The array has already been deep copied above.  This is by reference only.
    a[0:lp] = a[0:lp] * _calccostaper(lp)[::-1]
    a[-rp:] = a[-rp:] * _calccostaper(rp)
    return a


def _calccostaper(ntp):
    # Used by _costaper to compute a cosine taper from 1 to zero over
    # ntp points
    tp = np.zeros(ntp)
    for ii in range(1, ntp + 1):
        tp[ii - 1] = (1.0 + np.cos(ii*np.pi/ntp)/2) - 0.5
    return tp


def _nextpow2(ii):
    # Computes the next power of two
    buf = np.ceil(np.log(ii)/np.log(2))
    return int(2**buf)


def _is_number(s):
    # Returns true if s can be cast as a float, false otherwise
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_integer(s):
    # Returns true if s is an integer. Used for testing int/array
    try:
        int(s)
        return True
    except TypeError:
        return False
