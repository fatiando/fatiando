"""
Record the revision information, remove an old install, make a source
distribution and install it.
"""
import sys
from os.path import join
import subprocess

def setrevision():
    with open(join('fatiando','revision.py'), 'w') as revision:
        proc = subprocess.Popen('hg id', shell=True, stdout=subprocess.PIPE)
        revision.write("__revision__ = '%s'" % ("".join(proc.stdout.readlines()).strip()))

def uninstall():
    subprocess.Popen('echo "y" | pip uninstall fatiando', shell=True).wait()

def makedist():
    subprocess.Popen('python setup.py sdist', shell=True).wait()

def install():
    import fatiando
    distfile = 'fatiando-%s.tar.gz' % (fatiando.__version__)
    dist =join('dist', distfile)
    subprocess.Popen('pip install %s' % (dist), shell=True).wait()

def build_ext():
    """
    Build extention modules inplace.
    """
    subprocess.Popen('python setup.py build_ext --inplace', shell=True).wait()

if __name__ == '__main__':
    setrevision()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'install':
            uninstall()
            makedist()
            install()
        else:
            print "Unknown command:", sys.argv[1:]
            sys.exit()
    else:
        build_ext()
