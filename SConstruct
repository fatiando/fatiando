# Builder for the shared libs of the slooT package

import distutils.sysconfig
import os


# Phony target generator
def PhonyTarget(target, action, depends=[], env = None):

    if not env:
        env = DefaultEnvironment()

    env.Append(BUILDERS = { 'phony' : Builder(action = action)})

    phony = env.phony(target = target, source = 'SConstruct')

    Depends(phony, depends)

    env.AlwaysBuild(phony)


c_path = os.path.join('src', 'c')

wrap_path = os.path.join('src', 'wrap')


# DIRECT MODELS
################################################################################

# GRAVITY
gravity_outdir = os.path.join('fatiando', 'gravity')

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

# SEISMOLOGY
seismo_outdir = os.path.join('fatiando', 'seismo')

seismoenv = Environment(
    SWIGFLAGS=['-python', '-outdir', seismo_outdir],
    CPPPATH=[distutils.sysconfig.get_python_inc()],
    SHLIBPREFIX="")
    
traveltimemod = seismoenv.SharedLibrary( \
    target=os.path.join(seismo_outdir,'_traveltime'),
    source=[os.path.join(c_path, 'traveltime.c'), \
            os.path.join(wrap_path, 'traveltime.i')])

Depends(traveltimemod, os.path.join(c_path, 'traveltime.h'))

Clean(os.path.curdir, os.path.join(seismo_outdir,'traveltime.py'))
    
wavefdmod = seismoenv.SharedLibrary( \
    target=os.path.join(seismo_outdir,'_wavefd_ext'),
    source=[os.path.join(c_path, 'wavefd.c'), \
            os.path.join(wrap_path, 'wavefd.i')])

Depends(wavefdmod, os.path.join(c_path, 'wavefd.h'))

Clean(os.path.curdir, os.path.join(seismo_outdir,'wavefd_ext.py'))

# Group direct mods
directmods = [prismgravmod, traveltimemod, wavefdmod]

################################################################################


# Group all the C coded modules
ext_mods = []
ext_mods.extend(directmods)


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