"""
Pure Python implementations of functions in fatiando.gravmag.tesseroid. 
Used instead of Cython versions if those are not available.
"""
import numpy                                                                    
# Import Cython definitions for numpy                                           
cimport numpy                                                                   
                                                                                
DTYPE = numpy.float                                                             
ctypedef numpy.float_t DTYPE_T

__all__ = ['_need_to_divide']


def _need_to_divide(numpy.ndarray[DTYPE_T, ndim=1] distances, DTYPE_T size, 
    DTYPE_T ratio):
    """
    For which computation points the tesseroid must be divided.
    Based on the distances to the points and the distance/size ratio.
    """
    cdef unsigned int i, ndata = len(distances)
    result = []
    append = result.append
    for i in xrange(ndata):
        if distances[i] > 0 and distances[i] < ratio*size:
            append(i)
    return result 
