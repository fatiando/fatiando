"""
Test suite for the fatiando.math package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'

import unittest

import fatiando.math.tests.lu


def suite(label='fast'):

    suite = unittest.TestSuite()

    suite.addTest(fatiando.math.tests.lu.suite(label))

    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')