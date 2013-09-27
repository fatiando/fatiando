"""

"""
from scipy.special import hankel2, jn
import numpy

def wedge_cylindrical(rho, phi, rho_s, phi_s, c, source, dt):
    """
    Analytic solution equation (3) paper Alford et. al. cylindrical coordinates
    for \rho (observation point) smaller equal \rho_s (source point)
    (look figure 1. in paper)
    90 degrees wedge model cylindrical coordinates
    
    R.M. Alford - Accuracy of Finite-Difference Modeling

    rho - radius from center position of observation point
    phi - angle in radians position of observation point
    rho_s - radius from center position of source function
    phi_s - angle in radians position of source function
    c - velocity    
    source - source fuction sampled in time
    dt - sample rate from source function in time

    """
    N = len(source)
    dw = numpy.pi*2*(1./dt)/N # omega increment for each omega that will be sampled 
    ks = numpy.array([(p*dw/c) for p in xrange(N)]) # all k's = w/c in omega/frequency domain to evaluate the solution
    
    # serie aproximation just 10 first terms
    serie = numpy.zeros(N) +1j * numpy.zeros(N)
    for n in xrange(1,10):
        tmp = jn(2*n/3, ks*rho)*hankel2(2*n/3,ks*rho_s)*numpy.sin(2*n*phi_s/3)*numpy.sin(2*n*phi/3)
        tmp[0] = 0. # bessel and hankel undefined in 0
        serie += tmp    
    
    sourcew = numpy.fft.fft(source) # source in the frequency domain    
    return numpy.real(numpy.fft.ifft(numpy.complex(0, -8*numpy.pi/3) * sourcew * serie))


def free_1d(x, c, source, dt):
    """
    Analytic solution equation 1D, using green function for
    helmoltz wave equation and fourier transform.
    1D free space
    
    source at zero

    x - distance to source
    c - velocity
    source - source fuction sampled in time
    dt - sample rate from source function in time

    """
    
    N = len(source)
    dw = numpy.pi*2*(1./dt)/N # omega increment for each omega that will be sampled 
    ks = numpy.array([(p*dw/c) for p in xrange(N)]) # all k's = w/c in omega/frequency domain to evaluate the solution
    
    green = (0-1j)*(numpy.cos(ks*x)-1j*numpy.sin(ks*x))/(2*ks)    
    green[0] = 0.0  
    
    sourcew = numpy.fft.fft(source) # source in the frequency domain    
    return numpy.real(numpy.fft.ifft(green * sourcew))  


def free_2d(rho, c, source, dt):
    """
    Analytic solution equation Alford et. al. cylindrical coordinates
    2D free space
    
    source at zero
    
    note: dot not evaluate at zero, solution not viable
    
    rho - distance to the source function
    c - velocity
    source - source fuction sampled in time
    dt - sample rate from source function in time

    """
    N = len(source)
    dw = numpy.pi*2*(1./dt)/N # omega increment for each omega that will be sampled 
    ks = numpy.array([(p*dw/c) for p in xrange(N)]) # all k's = w/c in omega/frequency domain to evaluate the solution
    hankelshift = (-1j*numpy.pi)*hankel2(0,ks*rho)
    hankelshift[0] = 0. # i infinity in limit
    
    sourcew = numpy.fft.fft(source) # source in the frequency domain  
    return numpy.real(numpy.fft.ifft(hankelshift*sourcew)) 


def _2cylindrical(x, z, x0=0, z0=0):
    """
    Convert from cartezian coordinates to cylindrical coordinates
    x, z increses right and downward. The origin is (x0, z0)
    where x,z represents the R^2 such (x, z, y) = (i, j, k)
    
    !include better description of the two coordinate systems.!
    
    Returns:
    
    rho - radius
    phi - angle
    
    """
    # having z increasing upward would remove the -
    return ( numpy.sqrt((x-x0)**2+(z-z0)**2), (2*numpy.pi-numpy.arctan2(z-z0,x-x0))%(2*numpy.pi) )

def _2cartezian(rh, phi, rh0, phi0):
    pass





