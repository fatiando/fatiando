"""
Build the extension modules and documentation and run the test suite.

Run 'scons' to build and 'scons -c' to delete build files. 
"""

import distutils.sysconfig
import os


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


# Phony target generator
def PhonyTarget(target, action, depends=[], env = None):

    if not env:
        env = DefaultEnvironment()

    env.Append(BUILDERS = { 'phony' : Builder(action = action)})

    phony = env.phony(target = target, source = 'SConstruct')

    Depends(phony, depends)
    
    return phony

# DEFINE THE BASE PATHS
src_path = 'src'
c_path = os.path.join(src_path, 'c')
wrap_path = os.path.join(src_path, 'wrap')
doc_path = 'doc'
userguid_path = os.path.join(doc_path, 'userguide')
examples_path = 'examples'

# Build the extention modules with the setup.py script
target = 'build_ext'
action = 'python setup.py build_ext --inplace'
build_ext = PhonyTarget(target=target, action=action, depends=[])
env = DefaultEnvironment()
env.AlwaysBuild(build_ext)

Clean(os.path.curdir, os.path.join(wrap_path,'_prismmodule.c'))
Clean(os.path.curdir, os.path.join(wrap_path,'_traveltimemodule.c'))


# Build the documentation
target = 'build_doc'
action = 'make -C doc/userguide html'
build_doc = PhonyTarget(target=target, action=action, depends=[])
env = DefaultEnvironment()

Depends(build_doc, build_ext)

Clean(os.path.curdir, os.path.join(userguid_path, '_build'))


# Make a phony target for the tests (the fast test suite)
target = 'test'
action = 'python test.py -v'
test = PhonyTarget(target=target, action=action, depends=[])
env = DefaultEnvironment()
env.AlwaysBuild(test)

# Clean up the build
Clean(os.path.curdir, 'build')
Clean(os.path.curdir, 'dist')
Clean(os.path.curdir, 'MANIFEST')
Clean(os.path.curdir, list_ext(os.path.curdir, '.so'))
Clean(os.path.curdir, list_ext(os.path.curdir, '.pyc'))

# Clean up the example results
#Clean(os.path.curdir, list_ext(examples_path, '.pickle'))
#Clean(os.path.curdir, list_ext(examples_path, '.log'))
#Clean(os.path.curdir, list_ext(examples_path, '.txt'))
#Clean(os.path.curdir, list_ext(examples_path, '.png'))