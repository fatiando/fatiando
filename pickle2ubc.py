import pickle
import numpy

with open('mesh.pickle') as f:
    mesh = pickle.load(f)
    nz, ny, nx = mesh.shape
    def dummy(x):
        if x['value'] is None:
            return -100.
        return x['value']
    def ravel(m):
        nz, ny, nx = m.shape
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    yield m[k][j][i]
    numpy.savetxt("res.den", map(dummy, ravel(mesh)))
    with open("mesh.msh", 'w') as f:
        f.write("%d %d %d\n" % (ny, nx, nz))
        f.write("%g %g %g\n" % (mesh[0][0][0]['y1'], mesh[0][0][0]['x1'], -mesh[0][0][0]['z1']))
        f.write("%d*%g\n" % (ny, mesh[0][0][0]['y2'] - mesh[0][0][0]['y1']))
        f.write("%d*%g\n" % (nx, mesh[0][0][0]['x2'] - mesh[0][0][0]['x1']))
        f.write("%d*%g\n" % (nz, mesh[0][0][0]['z2'] - mesh[0][0][0]['z1']))
