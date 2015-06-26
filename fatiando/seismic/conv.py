"""
Zero-offset convolutional seismic modeling

Give a depth model and obtain a seismic zero-offset convolutional gather. You 
can give the wavelet, if you already have, or use one of the existing, from 
which we advise ricker wavelet (rickerwave function).

* :func:`~fatiando.seismic.conv.seismic_convolutional`: given the depth 
  velocity model and wavelet, it returns the convolutional seismic gather.
* :func:`~fatiando.seismic.conv.depth_2_time`: convert depth property model to 
  the model in time.  
* :func:`~fatiando.seismic.conv.rickerwave`: calculates a ricker wavelet.

**References**

Yilmaz, Oz,
Ch.2 Deconvolution. In: YILMAZ, Oz. Seismic Data Analysis: Processing, 
Inversion, and Interpretation of Seismic Data. Tulsa: Seg, 2001. Cap. 2. 
p. 159-270. Available at: <http://dx.doi.org/10.1190/1.9781560801580.ch2>


----

"""
from __future__ import division
import numpy as np
from scipy import interpolate  # linear interpolation of velocity/density


def seismic_convolutional_model(n_traces, vel_l, f, wavelet, dt=2.e-3, rho=1.):
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

    * n_traces: integers
        The vertical and horizontal grid dimensions
    * model : 2D-array
        Vp values
    * f : float
        Dominant frequency of the ricker wavelet
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram
    * wavelet : float
        The function to consider as source in the seismic modelling.

    Returns:

    * synth_l : 2D-array
        Resulting seismogram

    """
    # calculate RC
    rc = np.zeros(np.shape(vel_l))
    try:
    #dimension of rho must be the same of velocity grid, if both are matrix
        rc[1:, :] = ((vel_l[1:, :]*rho[1:, :]-vel_l[:-1, :]*rho[:-1, :]) /
                     (vel_l[1:, :]*rho[1:, :]+vel_l[:-1, :]*rho[:-1, :]))        
    except TypeError:
        rc[1:, :] = (vel_l[1:, :]-vel_l[:-1, :])/(vel_l[1:, :]+vel_l[:-1, :])
        
    w = wavelet(f, dt)
    # convolution
    synth_l = np.zeros(np.shape(rc))
    for j in range(0, n_traces):
        if np.shape(rc)[0] >= len(w):
            synth_l[:, j] = np.convolve(rc[:, j], w, mode='same')
        else:
            aux = np.floor(len(w)/2.)
            synth_l[:, j] = np.convolve(rc[:, j], w, mode='full')[aux:-aux]
    return synth_l

def depth_2_time(n_samples, n_traces, model, dt=2.e-3, dz=1., rho=1.):
    """
    Convert depth property model to time model.

    Parameters:
    * n_samples, n_traces: integers
        The vertical and horizontal grid dimensions
    * model : 2D-array
        Vp values
    * dt: float
        Sample time of the ricker wavelet and of the resulting seismogram
    * dz : float
        Length of square grid cells

    Returns:

    * TWT_ts : 1D-array
        Time axis for the property converted

    """
    #downsampled time rate to make a better interpolation
    dt_dwn = dt/10.
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
    vel_l=_resampling(model,TMAX,TWT,TWT_rs,dt,dt_dwn,n_traces)
    TWT_ts = np.zeros((np.ceil(TMAX/dt), n_traces))
    for j in range(1, len(TWT_ts)):
        TWT_ts[j, :] = TWT_rs[resmpl*j]
    # density calculations
    try:
       rho_l=_resampling(rho,TMAX,TWT,TWT_rs,dt,dt_dwn,n_traces)
    except TypeError:
        rho_l = rho
    return vel_l, rho_l

def _resampling(model,TMAX,TWT,TWT_rs,dt,dt_dwn,n_traces):
    """
    Resamples the input data to adjust it after time conversion with the chosen
    time sample rate, dt. 
    
    Returns:

    * vel_l : resampled input data
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
            vel_l[jj, j] = vel[resmpl*jj, j]  # 10=dt/dt_new, dt_new=0.002=2m
    return vel_l

def rickerwave(f, dt):
    """
    Given a frequency and time sampling rate, outputs ricker function.
    
    Parameters:

    * f : dominant frequency value in Hz
    * dt : time sampling rate in seconds (usually it is in the order of ms)

    Returns:

    * res : float
        ricker function for the given parameters
    
    """
    nw = 2.2/f/dt
    nw = 2*np.floor(nw/2)+1
    nc = np.floor(nw/2)
    result = np.zeros(nw)
    k = np.arange(1, nw+1)
    alpha = (nc-k+1)*f*dt*np.pi
    beta = alpha**2
    result = (1.-beta*2)*np.exp(-beta)
    return result
