import os
import re

def list_ext(path, ext):
    """List all the files witha given extention in path recursively"""
    files = []
    fnames = os.listdir(path)
    for fname in fnames:
        fname = os.path.join(path, fname)
        if os.path.isdir(fname):
            files.extend(list_ext(fname, ext))
        else:
            if os.path.splitext(fname)[-1] == ext:
                files.append(fname)
    return files

def findwrap(path):
    """Find the f2py generated wrappers"""
    iswrap = re.compile("module\.c$", re.IGNORECASE).search
    files = []
    fnames = os.listdir(path)
    for fname in fnames:
        fname = os.path.join(path, fname)
        if os.path.isdir(fname):
            files.extend(findwrap(fname))
        else:
            if iswrap(fname):
                files.append(fname)
    return files

# Clean up the build
Clean(os.path.curdir, 'build')
Clean(os.path.curdir, 'dist')
Clean(os.path.curdir, 'MANIFEST')
Clean(os.path.curdir, 'mylogfile.log')
Clean(os.path.curdir, os.path.join('fatiando', 'version.py'))
Clean(os.path.curdir, findwrap('src'))
Clean(os.path.curdir, list_ext(os.path.curdir, '.so'))
Clean(os.path.curdir, list_ext(os.path.curdir, '.pyc'))
