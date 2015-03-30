"""

"""

import numpy as np
def seismic_convolutional_model(n_samples, n_traces,model,f,dz=1.,dt=2.e-3,rho=1.):
    """
    Calculate the amplitude of the analytic signal of the data.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units and so will the analytic signal! I strongly recommend
        converting the data to SI **before** calculating the derivative (use
        one of the unit conversion functions of :mod:`fatiando.utils`).

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * ansig : 1D-array
        The amplitude of the analytic signal
    """
    
    
    dt_dwn=dt/10.
    
    TWT=np.zeros((n_samples, n_traces));
    TWT[0,:]=2*dz/model[0,:]
    for j in range(1,n_samples):
        TWT[j,:]=TWT[j-1]+2*dz/model[j,:]
    TMAX=max(TWT[-1,:])
    ######
    TMIN=min(TWT[0,:])
#    if dt/TMIN
    
    
    TWT_rs=np.zeros(np.ceil(TMAX/dt_dwn))
    for j in range(1,len(TWT_rs)):
        TWT_rs[j]=TWT_rs[j-1]+dt_dwn;
        
    #linear interpolation of velocity/density
    from scipy import interpolate

    vel=np.ones((np.ceil(TMAX/dt_dwn), n_traces))

    for j in range(0,n_traces):
        kk=np.ceil(TWT[0,j]/dt_dwn)
        lim=np.ceil(TWT[-1,j]/dt_dwn)-1;
    #linear interpolation
        tck=interpolate.interp1d(TWT[:,j],model[:,j])
        vel[kk:lim,j]=tck(TWT_rs[kk:lim])
        vel[lim:,j]=vel[lim-1,j]          #extension of the model repeats the last value of the true model
        vel[0:kk,j]=model[0,j]            #first values equal to the first value of the true model 
             
    #resampling from dt_dwn to dt
    vel_l=np.zeros((np.ceil(TMAX/dt),n_traces))
    TWT_ts=np.zeros((np.ceil(TMAX/dt),n_traces))

    resmpl=int(dt/dt_dwn)

    vel_l[0,:]=vel[0,:]
    for j in range(0,n_traces):
        for jj in range(1,len(TWT_ts)):
            vel_l[jj,j]=vel[resmpl*jj,j] #10=dt/dt_new, dt_new=0.002=2ms

    for j in range(1,len(TWT_ts)):
        TWT_ts[j,:]=TWT_rs[resmpl*j]
    #calculate RC
    rc=np.zeros(np.shape(vel_l));
    rc[1:,:]=(vel_l[1:,:]-vel_l[:-1,:])/(vel_l[1:,:]+vel_l[:-1,:])

    #####
    ##wavelet - put as another function
    nw=2.2/f/dt;
    nw=2*np.floor(nw/2)+1;

    nc=np.floor(nw/2);
    w = np.zeros(nw);
    k=np.arange(1,nw+1);

    alpha = (nc-k+1)*f*dt*np.pi;
    beta=alpha**2;
    w = (1.-beta*2)*np.exp(-beta);
    ##
    
    ###convolution
    synth_l=np.zeros(np.shape(rc))
    for j in range(0,n_traces):
        if np.shape(rc)[0]>=len(w):
            synth_l[:,j]=np.convolve(rc[:,j],w,mode='same')
        else:
            synth_l[:,j]=np.convolve(rc[:,j],w,mode='full')[np.floor(len(w)/2.):-np.floor(len(w)/2.)]

    
    return synth_l,TWT_ts
