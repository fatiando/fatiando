"""
Test suite for the fatiando.math package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'

import unittest

import fatiando.math.tests.lu
import fatiando.math.tests.glq


def suite(label='fast'):

    testsuite = unittest.TestSuite()

    testsuite.addTest(fatiando.math.tests.lu.suite(label))
    
    testsuite.addTest(fatiando.math.tests.glq.suite(label))

    return testsuite


if __name__ == '__main__':
    
    unittest.main(defaultTest='suite')