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
        Reflectivity values in the time domain.
    * f : float
        Dominant frequency of the ricker wavelet.
    * wavelet : float
        The function to consider as source in the seismic modelling.
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram, in
        general a value of 2.e-3 is used.

    Returns:

    * synth_l : 2D-array
        Resulting seismogram.

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


def reflectivity(model_t, rho):
    """
    Calculate reflectivity series in the time domain, so it is necessary to use
    the function depth_2_time first if the model is in depth domain. Note this
    this function can also be used to one dimensional array.

    Parameters:

    * model_t : 2D-array
        Velocity values in time domain.
    * rho : 2D-array (optional)
        Density values for all the model, in time domain.

    Returns:

    * rc : 2D-array
        Calculated reflectivity series for all the model given.

    """
    err_message = "Velocity and density matrix must have the same dimension."
    assert model_t.shape == rho.shape, err_message
    rc = np.zeros(np.shape(model_t))
    rc[1:, :] = ((model_t[1:]*rho[1:]-model_t[:-1]*rho[:-1]) /
                 (model_t[1:]*rho[1:]+model_t[:-1]*rho[:-1]))
    return rc


def depth_2_time(vel, model, dt, dz):
    """
    Convert depth property model to time model.

    Parameters:

    * vel : 2D-array
        Velocity values in the depth domain.
    * model : 2D-array
        Model values in the depth domain.
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram, in
        general a value of 2.e-3 is used.
    * dz : float
        Length of square grid cells.
    * rho : 2D-array (optional)
        Density values for all the model, in depth domain.

    Returns:

    * model_t : 2D-array
        Property model in time domain.

    """
    err_message = "Velocity and model matrix must have the same dimension."
    assert vel.shape == model.shape, err_message
    # downsampled time rate to make a better interpolation
    n_samples, n_traces = [vel.shape[0], vel.shape[1]]
    dt_dwn = dt/10.
    if dt_dwn > dz/np.max(vel):
        dt_dwn = (dz/np.max(vel))/10.
    TWT = np.zeros((n_samples, n_traces))
    TWT[0, :] = 2*dz/vel[0, :]
    for j in range(1, n_samples):
        TWT[j, :] = TWT[j-1]+2*dz/vel[j, :]
    TMAX = max(TWT[-1, :])
    TMIN = min(TWT[0, :])
    TWT_rs = np.zeros(np.ceil(TMAX/dt_dwn))
    for j in range(1, len(TWT_rs)):
        TWT_rs[j] = TWT_rs[j-1]+dt_dwn
    resmpl = int(dt/dt_dwn)
    model_t = _resampling(model, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces)
    return model_t


def _resampling(model, TMAX, TWT, TWT_rs, dt, dt_dwn, n_traces):
    """
    Resamples the input data to adjust it after time conversion with the chosen
    time sample rate, dt.

    Returns:

    * vel_l : 2D-array
        Resampled input data.

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
        Ricker function for the given parameters.

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
