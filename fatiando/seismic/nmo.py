# coding: utf-8
"""
"""
from __future__ import division, print_function, unicode_literals
from future.builtins import range

import numpy as np
try:
    from scipy.interpolate import CubicSpline
except ImportError:
    from scipy.interpolate import interp1d
    CubicSpline = None

from .._our_duecredit import due, Doi


@due.dcite(Doi("10.1190/1.9781560801580"),
           description='Seismic Data Analysis book')
def nmo_correction(cmp, dt, offsets, velocities):
    """
    Performs NMO correction on the given CMP.

    The units must be consistent. E.g., if dt is seconds and
    offsets is meters, velocities must be m/s.

    Uses the method described in Yilmaz (2001).

    Parameters:

    cmp : 2D array
        The CMP gather that we want to correct.
    dt : float
        The sampling interval.
    offsets : 1D array
        An array with the offset of each trace in the CMP.
    velocities : 1D array
        An array with the NMO velocity for each time. Should
        have the same number of elements as the CMP has samples.

    Returns:

    nmo : 2D array
        The NMO corrected gather.

    References:

    Yilmaz, Ö. (2001), Seismic Data Analysis: Processing, Inversion, and
    Interpretation of Seismic Data, Society of Exploration Geophysicists.
    doi:10.1190/1.9781560801580

    """
    nmo = np.zeros_like(cmp)
    nsamples = cmp.shape[0]
    times = np.arange(0, nsamples*dt, dt)
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            amplitude = sample_trace(cmp[:, j], t, dt)
            # If the time t is outside of the CMP time range,
            # amplitude will be None.
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo
