from __future__ import absolute_import
import os
from pytest import raises

from .. import check_hash

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_check_hash():
    "Make sure check_hash works for the Surfer test data"
    fname = os.path.join(TEST_DATA_DIR, 'simple_surfer.grd')
    # Hashes gotten from openssl
    sha256 = "9cbdae1c020797231ff45a18594f80c68c3147d0b976103767a0c2c333b07ff6"
    check_hash(fname, sha256, hash_type='sha256')
    md5 = '70e2e6f0f37fba97a3545fcab8ffab21'
    check_hash(fname, md5, hash_type='md5')


def test_check_hash_fails():
    "Test if check_hash fails properly for a wrong known hash"
    fname = os.path.join(TEST_DATA_DIR, 'simple_surfer.grd')
    # Hashes gotten from openssl and changed by a single character
    sha256 = "acbdae1c020797231ff45a18594f80c68c3147d0b976103767a0c2c333b07ff6"
    with raises(AssertionError):
        check_hash(fname, sha256, hash_type='sha256')
    md5 = 'a0e2e6f0f37fba97a3545fcab8ffab21'
    with raises(AssertionError):
        check_hash(fname, md5, hash_type='md5')
