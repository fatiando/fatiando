"""
Pure Python implementations of functions in fatiando.gravmag.tesseroid. 
Used instead of Cython versions if those are not available.
"""

__all__ = ['_need_to_divide']

def _need_to_divide(distances, size, ratio):
    """
    For which computation points the tesseroid must be divided.
    Based on the distances to the points and the distance/size ratio.
    """
    return [i for i in xrange(len(distances))                                 
            if distances[i] > 0 and distances[i] < ratio*size]
