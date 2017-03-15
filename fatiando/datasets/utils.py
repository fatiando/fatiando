"""
Utilities for loading the data sets.
"""
from __future__ import absolute_import
import hashlib


def check_hash(fname, known_hash, hash_type='sha256'):
    """
    Assert that the hash of a file is equal to a known hash.

    Useful for checking if a file has changed or been corrupted.

    Parameters:

    * fname : string
        The name of the file.
    * known_hash : string
        The known (recorded) hash of the file.
    * hash_type : string
        What kind of hash is it. Can be anything defined in Python's hashlib.

    Raises:

    * ``AssertionError`` if the hash is different from the known hash.

    """
    # Calculate the hash in chunks to avoid overloading the memory
    chunksize = 65536
    hasher = getattr(hashlib, hash_type)()
    with open(fname, 'rb') as f:
        buf = f.read(chunksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(chunksize)
    file_hash = hasher.hexdigest()
    msg = '\n'.join([
        'Possibly corrupted file {}.'.format(fname),
        '  - Calculated {} hash: {}'.format(hash_type, file_hash),
        '  - Known (recorded) hash: {}'.format(known_hash)])
    assert file_hash == known_hash, msg
