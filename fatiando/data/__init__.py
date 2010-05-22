"""
fatiando.data:
    Collection of containers (classes) for different geophysical data. Each 
    class has methods for loading from files, generating synthetic data, etc.
    The specific data types all inherit from the GeoData class and implement
    the 'array', and 'cov' properties.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 21-May-2010'

    
import numpy


def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando.data package.

    Parameters:

        label: can be either 'fast' for a smaller and faster test
               or 'full' for the full test suite

        verbose: controls if the whole test information is printed
                 or just the final results
    """
    
    if label != 'fast' and label != 'full':
        
        from exceptions import ValueError
        
        raise ValueError("Test label must be either 'fast' or 'full'")

    import unittest

    import tests

    suite = unittest.TestSuite()
    
    suite.addTest(tests.suite(label))    

    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    runner.run(suite)
    
    
    
    
class GeoData():
    """
    Base class for holding geophysical data.
    """
    
    def __init__(self):
        pass
    
    
    def __len__(self):
        
        return 0
    
    
    def _toarray(self):
        """
        Convert the data (only value, no position) to a Numpy array.
        """
        raise NotImplementedError, "Tried to call _toarray (or array" + \
            " property) without implementation."
    
    
    array = property(_toarray)
    
    
    def _get_cov(self):
        """
        Build the covariance matrix based on the standard deviation values.
        """
        
        raise NotImplementedError, "Tried to call _get_cov (or cov" + \
            " property) without implementation."
    
    
    cov = property(_get_cov)
    
    