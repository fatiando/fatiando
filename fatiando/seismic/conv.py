"""
Zero-offset convolutional seismic modeling

Give a depth model and obtain a seismic zero-offset convolutional gather. You
can give the wavelet, if you already have, or use one of the existing, from
which we advise ricker wavelet (rickerwave function).

* :func:`~fatiando.seismic.conv.convolutional_model`: given the reflectivity
  series and wavelet, it returns the convolutional seismic gather.
* :func:`~fatiando.seismic.conv.reflectivity`: calculates the reflectivity
  series from the velocity model (and density model if present).
* :func:`~fatiando.seismic.conv.depth_2_time`: convert depth property model to
  the model in time.
* :func:`~fatiando.seismic.conv.rickerwave`: calculates a ricker wavelet.

Examples
--------
.. plot::
    :include-source:
    :context:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from fatiando.seismic import conv
    >>> from fatiando.vis import mpl
    >>> # Choose some velocity depth model
    >>> n_samples, n_traces = [600, 20]
    >>> rock_grid = 1500.*np.ones((n_samples, n_traces))
    >>> rock_grid[300:, :] = 2500.
    >>> # Convert from depth to time
    >>> [vel_l, rho_l] = conv.depth_2_time(rock_grid, dt=2.e-3, dz=1.)
    >>> # Calculate the reflectivity for all the model
    >>> rc = conv.reflectivity(vel_l, rho_l)
    >>> # Convolve the reflectivity with a ricker wavelet
    >>> synt = conv.convolutional_model(rc, 30., conv.rickerwave, dt=2.e-3)
    >>> # Plot the result
    >>> fig = plt.figure(figsize=(6,5))
    >>> _ = mpl.seismic_wiggle(synt, dt=2.e-3)
    >>> _ = mpl.seismic_image(synt, dt=2.e-3,
    ...                            cmap=mpl.pyplot.cm.jet, aspect='auto')
    >>> _ = plt.ylabel('time (seconds)')
    >>> _ = plt.title("Convolutional seismogram", fontsize=13, family='sans-serif')

References
----------

Yilmaz, Oz,
Ch.2 Deconvolution. In: YILMAZ, Oz. Seismic Data Analysis: Processing,
Inversion, and Interpretation of Seismic Data. Tulsa: Seg, 2001. Cap. 2.
p. 159-270. Available at: <http://dx.doi.org/10.1190/1.9781560801580.ch2>


"""
from __future__ import division
import numpy as np
from scipy import interpolate  # linear interpolation of velocity/density


def convolutional_model(rc, f, wavelet, dt):
    """
    Calculate convolutional seismogram for a geological model

    Calculate the synthetic convolutional seismogram of a geological model, Vp
    is mandatory while density is optional. The given model in a matrix form is
    considered a mesh of square cells.

    .. warning::

        Since the relative difference between the model is the important, being
        consistent with the units chosen for the parameters is the only
        requirement, whatever the units.

    Parameters:

    * rc : 2D-array
        reflectivity values in the time domain
    * f : float
        Dominant frequency of the ricker wavelet
    * wavelet : float
        The function to consider as source in the seismic modelling.
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram, in 
        general a value of 2.e-3 is used.

    Returns:

    * synth_l : 2D-array
        Resulting seismogram

    """
    w = wavelet(f, dt)
    # convolution
    synth_l = np.zeros(np.shape(rc))
    for j in range(0, rc.shape[1]):
        if np.shape(rc)[0] >= len(w):
            synth_l[:, j] = np.convolve(rc[:, j], w, mode='same')
        else:
            aux = np.floor(len(w)/2.)
            synth_l[:, j] = np.convolve(rc[:, j], w, mode='full')[aux:-aux]
    return synth_l


def reflectivity(model_t, rho=1.):
    """
    Calculate reflectivity series in the time domain, so it is necessary to use
    the function depth_2_time first if the model is in depth domain. 

    Parameters:

    * model_t : 2D-array
        Vp values in time domain
    * rho : 2D-array (optional)
        Density values for all the model, in time domain.

    Returns:

    * rc : 2D-array
        Calculated reflectivity series for all the model given.
    """
    rc = np.zeros(np.shape(model_t))
    # dimension of rho must be the same of velocity grid, if both are matrix
    try:
        rc[1:, :] = ((model_t[1:, :]*rho[1:, :]-model_t[:-1, :]*rho[:-1, :]) /
                     (model_t[1:, :]*rho[1:, :]+model_t[:-1, :]*rho[:-1, :]))
    except TypeError:
        rc[1:, :] = ((model_t[1:, :]-model_t[:-1, :]) /
                     (model_t[1:, :]+model_t[:-1, :]))
    return rc


def depth_2_time(model, dt, dz, rho=1.0):
    """
    Convert depth property model to time model.

    Parameters:

    * model : 2D-array
        Vp values in the depth domain.
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram, in 
        general a value of 2.e-3 is used.
    * dz : float
        Length of square grid cells
    * rho : 2D-array (optional)
        Density values for all the model, in depth domain.

    Returns:

    * vel_l : 2D-array
        Velocity model in time domain
    * rho_l : 2D-array
        Density model in time domain

    """
    # downsampled time rate to make a better interpolation
    n_samples, n_traces = [model.shape[0], model.shape[1]]
    dt_dwn = dt/10.
    if dt_dwn > dz/np.max(model):
        dt_dwn = (dz/np.max(model))/10.
    TWT = np.zeros((n_samples, n_traces))
    TWT[0, :] = 2*dz/model[0, :]
    for j in range(1, n_samples):
        TWT[j, :] = TWT[j-1]+2*dz/model[j, :]
    TMAX = max(TWT[-1, :])
    TMIN = min(TWT[0, :])
    TWT_rs = np.zeros(np.ceil(TMAX/dt_dwn))
    for j in range(1, len(TWT_rs)):
        TWT_rs[j] = TWT_rs[j-1]+dt_dwn
    resmpl = int(dt/dt_dwn)
    vel_l = _resampling(model, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces)
    TWT_ts = np.zeros((np.ceil(TMAX/dt), n_traces))
    for j in range(1, len(TWT_ts)):
        TWT_ts[j, :] = TWT_rs[resmpl*j]
    # density calculations
    try:
        rho_l = _resampling(rho, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces)
    except TypeError:
        rho_l = rho
    return vel_l, rho_l


def _resampling(model, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces):
    """
    Resamples the input data to adjust it after time conversion with the chosen
    time sample rate, dt.

    Returns:

    * vel_l : 2D-array
        resampled input data

    """
    vel = np.ones((np.ceil(TMAX/dt_dwn), n_traces))
    for j in range(0, n_traces):
        kk = np.ceil(TWT[0, j]/dt_dwn)
        lim = np.ceil(TWT[-1, j]/dt_dwn)-1
    # necessary do before resampling to have values in all points of time model
        tck = interpolate.interp1d(TWT[:, j], model[:, j])
        vel[kk:lim, j] = tck(TWT_rs[kk:lim])
    # the model is extended in time because of depth time conversion
        vel[lim:, j] = vel[lim-1, j]
    # because of time conversion, the values between 0 e kk need to be filed
        vel[0:kk, j] = model[0, j]
    # resampling from dt_dwn to dt
    vel_l = np.zeros((np.ceil(TMAX/dt), n_traces))
    resmpl = int(dt/dt_dwn)
    vel_l[0, :] = vel[0, :]
    for j in range(0, n_traces):
        for jj in range(1, int(np.ceil(TMAX/dt))):
            vel_l[jj, j] = vel[resmpl*jj, j]
    return vel_l


def rickerwave(f, dt):
    r"""
    Given a frequency and time sampling rate, outputs ricker function. The
    length of the function varies according to f and dt, in order for the
    ricker function starts and ends as zero. It is also considered that the
    functions is causal, what means it starts at time zero. To satisfy sampling
    and stability:

    .. math::

        f << \frac{1}{2 dt}.

    Here, we consider this as:

    .. math::

        f < 0.2 \frac{1}{2 dt}.

    Parameters:

    * f : float
        dominant frequency value in Hz
    * dt : float
        time sampling rate in seconds (usually it is in the order of ms)

    Returns:

    * ricker : float
        ricker function for the given parameters

    """
    assert f < 0.2*(1./(2.*dt)), "Frequency too high for the dt chosen."
    nw = 2.2/f/dt
    nw = 2*np.floor(nw/2)+1
    nc = np.floor(nw/2)
    ricker = np.zeros(nw)
    k = np.arange(1, nw+1)
    alpha = (nc-k+1)*f*dt*np.pi
    beta = alpha**2
    ricker = (1.-beta*2)*np.exp(-beta)
    return ricker
