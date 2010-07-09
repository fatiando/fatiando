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

c_path = os.path.join('src', 'c')
wrap_path = os.path.join('src', 'wrap')

# MATH
################################################################################
math_outdir = os.path.join('fatiando', 'math')

mathenv = Environment(
    SWIGFLAGS=['-python', '-outdir', math_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")

lumod = mathenv.SharedLibrary(target=os.path.join(math_outdir,'_lu'),
    source=[os.path.join(c_path, 'lu.c'), os.path.join(wrap_path, 'lu.i')])

Depends(lumod, os.path.join(c_path, 'lu.h'))

Clean(os.path.curdir,os.path.join(math_outdir,'lu.py'))

glqmod = mathenv.SharedLibrary(target=os.path.join(math_outdir,'_glq'),
  source=[os.path.join(c_path, 'glq.c'), os.path.join(wrap_path, 'glq.i')])

Depends(glqmod, os.path.join(c_path, 'glq.h'))

Clean(os.path.curdir,os.path.join(math_outdir,'glq.py'))

# Group the math mods
mathmods = [lumod, glqmod]

################################################################################


# DIRECT MODELS
################################################################################
direct_outdir = os.path.join('fatiando', 'directmodels')

# GRAVITY 
###########################################
gravity_outdir = os.path.join(direct_outdir, 'gravity')

gravityenv = Environment(
    SWIGFLAGS=['-python', '-outdir', gravity_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")

prismgravmod = gravityenv.SharedLibrary( \
    target=os.path.join(gravity_outdir,'_prism'),
    source=[os.path.join(c_path, 'prismgrav.c'), \
            os.path.join(wrap_path, 'prismgrav.i')])
            
Depends(prismgravmod, os.path.join(c_path, 'prismgrav.h'))

Clean(os.path.curdir, os.path.join(gravity_outdir,'prism.py'))

tesseroidgravmod = gravityenv.SharedLibrary( \
    target=os.path.join(gravity_outdir,'_tesseroid'),
    source=[os.path.join(c_path, 'tesseroidgrav.c'), \
            os.path.join(wrap_path, 'tesseroidgrav.i')])
            
Depends(tesseroidgravmod, os.path.join(c_path, 'tesseroidgrav.h'))

Clean(os.path.curdir, os.path.join(gravity_outdir,'tesseroid.py'))
###########################################

# SEISMO
###########################################
seismo_outdir = os.path.join(direct_outdir, 'seismo')

seismoenv = Environment(
    SWIGFLAGS=['-python', '-outdir', seismo_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")
    
simpletommod = seismoenv.SharedLibrary( \
    target=os.path.join(seismo_outdir,'_simple'),
    source=[os.path.join(c_path, 'simpletom.c'), \
            os.path.join(wrap_path, 'simpletom.i')])

Depends(simpletommod, os.path.join(c_path, 'simpletom.h'))

Clean(os.path.curdir, os.path.join(seismo_outdir,'simple.py'))
    
wavefdmod = seismoenv.SharedLibrary( \
    target=os.path.join(seismo_outdir,'_wavefd_ext'),
    source=[os.path.join(c_path, 'wavefd.c'), \
            os.path.join(wrap_path, 'wavefd.i')])

Depends(wavefdmod, os.path.join(c_path, 'wavefd.h'))

Clean(os.path.curdir, os.path.join(seismo_outdir,'wavefd_ext.py'))
###########################################

# Group direct mods
directmods = [prismgravmod, simpletommod, wavefdmod]

################################################################################


# GEOMETRY
################################################################################
geometry_outdir = os.path.join('fatiando', 'utils')

geometryenv = Environment(
    SWIGFLAGS=['-python', '-shadow', '-outdir', geometry_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")
    
geometrymod = geometryenv.SharedLibrary( \
    target=os.path.join(geometry_outdir ,'_geometry'),
    source=[os.path.join(wrap_path, 'geometry.i')])
        

Depends(geometrymod, os.path.join(c_path, 'geometry.c'))

Clean(os.path.curdir, os.path.join(geometry_outdir,'geometry.py'))
################################################################################

# Group all the C coded modules
ext_mods = []
ext_mods.extend(directmods)
ext_mods.extend(mathmods)

################################################################################
################################################################################


# Make a phony target for the tests (the fast test suite)
target = 'unittest_runner'
action = 'python %s.py -v' % (target)
PhonyTarget(target=target, action=action, depends=ext_mods)


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
