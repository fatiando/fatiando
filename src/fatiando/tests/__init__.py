"""
Test suite for the fatiando package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Mar-2010'

import unittest

import fatiando.directmodels.tests
import fatiando.geoinv.tests
import fatiando.math.tests
import fatiando.utils.tests
import fatiando.data.tests


def suite(label='fast'):

    testsuite = unittest.TestSuite()

    testsuite.addTest(fatiando.directmodels.tests.suite(label))
    testsuite.addTest(fatiando.geoinv.tests.suite(label))
    testsuite.addTest(fatiando.math.tests.suite(label))
    testsuite.addTest(fatiando.utils.tests.suite(label))
    testsuite.addTest(fatiando.data.tests.suite(label))

    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')