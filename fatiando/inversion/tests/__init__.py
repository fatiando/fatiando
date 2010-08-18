"""
Test suite for the fatiando.inversion package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'

import unittest

def suite(label='fast'):

    testsuite = unittest.TestSuite()

    return testsuite


if __name__ == '__main__':
    
    unittest.main(defaultTest='suite')