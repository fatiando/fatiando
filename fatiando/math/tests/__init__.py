"""
Test suite for the fatiando.math package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'

import unittest

from fatiando.math.tests import lu, glq


def suite(label='fast'):

    testsuite = unittest.TestSuite()

    testsuite.addTest(lu.suite(label))
    
    testsuite.addTest(glq.suite(label))

    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')