# Builder for the shared libs of the slooT package

import distutils.sysconfig
import os


# Phony target generator
################################################################################
def PhonyTarget(target, action, depends=[], env = None):

    if not env:
        env = DefaultEnvironment()

    env.Append(BUILDERS = { 'phony' : Builder(action = action)})

    phony = env.phony(target = target, source = 'SConstruct')

    Depends(phony, depends)

    env.AlwaysBuild(phony)
################################################################################


# PYTHON EXTENTIONS (SWIG)
################################################################################
################################################################################

c_path = 'c'

# MATH
################################################################################
cmath_path = os.path.join(c_path, 'math')
math_outdir = 'fatiando/math'

mathenv = Environment(
    SWIGFLAGS=['-python', '-outdir', math_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")

lumod = mathenv.SharedLibrary(os.path.join(math_outdir,'_lu'),
    [os.path.join(cmath_path, 'lu.c'), os.path.join(cmath_path, 'lu.i')])

Depends(lumod, os.path.join(cmath_path, 'lu.h'))

Clean(os.path.curdir,os.path.join(math_outdir,'lu.py'))

glqmod = mathenv.SharedLibrary(os.path.join(math_outdir,'_glq'),
    [os.path.join(cmath_path, 'glq.c'), os.path.join(cmath_path, 'glq.i')])

Depends(glqmod, os.path.join(cmath_path, 'glq.h'))

Clean(os.path.curdir,os.path.join(math_outdir,'glq.py'))

# Group the math mods
mathmods = [lumod, glqmod]

################################################################################


# DIRECT MODELS
################################################################################
direct_outdir = 'fatiando/directmodels'
cdirect_path = os.path.join(c_path, 'directmodels')

# GRAVITY 
###########################################
gravity_outdir = os.path.join(direct_outdir, 'gravity')
gravityenv = Environment(
    SWIGFLAGS=['-python', '-outdir', gravity_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")

prismgravmod = gravityenv.SharedLibrary(os.path.join(gravity_outdir,'_prism'),
    [os.path.join(cdirect_path, 'prismgrav.c'), \
     os.path.join(cdirect_path, 'prismgrav.i')])

Depends(prismgravmod, os.path.join(cdirect_path, 'prismgrav.h'))

Clean(os.path.curdir,os.path.join(gravity_outdir,'prism.py'))
###########################################

# SEISMO
###########################################
seismo_outdir = os.path.join(direct_outdir, 'seismo')
seismoenv = Environment(
    SWIGFLAGS=['-python', '-outdir', seismo_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")
    
simpletommod = seismoenv.SharedLibrary(os.path.join(seismo_outdir,'_simple'),
    [os.path.join(cdirect_path, 'simpletom.c'), \
     os.path.join(cdirect_path, 'simpletom.i')])

Depends(simpletommod, os.path.join(cdirect_path, 'simpletom.h'))

Clean(os.path.curdir,os.path.join(seismo_outdir,'simple.py'))
###########################################

# Group direct mods
directmods = [prismgravmod, simpletommod]

################################################################################


# Group all the C coded modules
ext_mods = [directmods, mathmods]

################################################################################
################################################################################


# Make a phony target for the tests (the fast test suite)
PhonyTarget(target='test', action='python test.py -v', depends=ext_mods)

# An alias for running the full test suite
full_test = Alias('test_full', ext_mods, 'python test.py -v -full')
AlwaysBuild(full_test)


# Include all the .pyc files in the cleaning
def listpyc(path):
    """List all the .pyc files in path recursively"""

    pycfiles = []

    fnames = os.listdir(path)
    for fname in fnames:

        fname = os.path.join(path, fname)

        if os.path.isdir(fname):
            pycfiles.extend(listpyc(fname))

        else:
            if os.path.splitext(fname)[-1] == '.pyc':
                pycfiles.append(fname)

    return pycfiles

Clean(os.path.curdir, listpyc(os.path.curdir))