"""
ClimateSignal:
    Direct models for climate change heat propagation into the Earth.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 19-Apr-2010'


import math
import numpy


seconds2years = 365.25*24*3600

def abrupt(depth, magnitude, time, thermal_diff=31.557599999999997, \
           step=0.001):
    """
    Calculates the effect at depth of an abrupt change in temperature at the 
    surface that happened at a past time.
    
    Parameters:
    
        depth: depth at which you want to calculate the effect
        
        magnitude: magnitude of the change in temperature
        
        time: how long ago the change happened
        
        thermal_diff: thermal property of the rocks
        
        step: size of the rectangles used in the integration rule
    """
    
    # Evaluate the erfc(x) = 1 - erf(x) function using a rectangular rule
    x = depth/math.sqrt(4*thermal_diff*time)
    
    erf = 0.0
    
    for xsi in numpy.arange(0.5*step, x, step):
        
        erf += math.exp(-(xsi**2))*step        
        
    erf *= 2./math.sqrt(math.pi)
    
    return magnitude*(1 - erf)


def abrupt_time_derivative(depth, magnitude, time, \
                           thermal_diff=31.557599999999997):
    """
    The first derivative of the abrupt change in relation to time.
    """
    x = depth/math.sqrt(4*thermal_diff*time)
    
    deriv = magnitude*math.exp(-(x**2))*x/ \
            (math.sqrt(math.pi)*time)
    
    return deriv


def linear(depth, magnitude, time, thermal_diff=31.557599999999997, \
           step=0.001):
    """
    Calculates the effect at depth of a linear change in temperature at the 
    surface that happened at a past time.
    
    Parameters:
    
        depth: depth at which you want to calculate the effect
        
        magnitude: magnitude of the change in temperature
        
        time: how long ago the change happened
        
        thermal_diff: thermal property of the rocks 
        
        step: size of the rectangles used in the integration rule
    """
    
    # Evaluate the erfc(x) = 1 - erf(x) function using a rectangular rule
    x = depth/math.sqrt(4*thermal_diff*time)
    
    erf = 0.0
    
    for xsi in numpy.arange(0.5*step, x, step):
        
        erf += math.exp(-(xsi**2))*step        
        
    erf *= 2./math.sqrt(math.pi)
    
    erfc = 1 - erf
    
    signal = magnitude*((1 + 2*(x**2))*erfc - \
                        x*2*math.exp(-(x**2))/math.sqrt(math.pi))
    
    return signal

    
    
    