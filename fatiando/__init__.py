"""
fatiando:
    Geophysical direct and inverse modeling package. Includes various direct
    models, inversion programs, a useful math package and various utils for 
    general geophysics tasks.
"""

__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'


# Set the default handler to the class loggers. 
# By default, logging is set to stderr.
################################################################################ 
import logging
import sys

# Create console handler and set level to debug
default_log_handler = logging.StreamHandler(strm=sys.stderr)

# Create formatter
formatter = logging.Formatter("%(name)s> %(message)s")

# Add formatter to the console handler
default_log_handler.setFormatter(formatter)

# Set verbose level
default_log_handler.setLevel(logging.INFO)
################################################################################


def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando package.

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