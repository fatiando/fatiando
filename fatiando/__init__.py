"""
fatiando:
    Geophysical direct and inverse modeling package. Includes various direct
    models, inversion programs, a useful math package and various utils for 
    general geophysics tasks.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'


def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando package.

    Parameters:

        label: can be either 'fast' for a smaller and faster test
               or 'full' for the full test suite

        verbose: controls if the whole test information is printed
                 or just the final results
    """
    if label!='fast' and label!='full':
        from exceptions import ValueError
        raise ValueError("Test label must be either 'fast' or 'full'")

    import unittest

    import fatiando.tests

    suite = unittest.TestSuite()
    suite.addTest(fatiando.tests.suite(label))

    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    runner.run(suite)