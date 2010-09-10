"""
Test suite for the fatiando package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Mar-2010'

import unittest

import fatiando.gravity.tests
import fatiando.seismo.tests
import fatiando.inversion.tests


def suite(label='fast'):

    testsuite = unittest.TestSuite()

    testsuite.addTest(fatiando.gravity.tests.suite(label))
    testsuite.addTest(fatiando.seismo.tests.suite(label))
    testsuite.addTest(fatiando.inversion.tests.suite(label))

    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')