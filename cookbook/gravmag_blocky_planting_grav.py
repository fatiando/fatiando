"""
GravMag: 3D blocky gravity inversion by planting anomalous densities
"""
from fatiando import gridder, utils
from fatiando.gravmag import polyprism
from fatiando.gravmag.blocky import sow, Gravity3D
from fatiando.mesher import PolygonalPrism, PrismMesh, vremove
from fatiando.vis import mpl, myv

# Create a synthetic model
bounds = [-10000, 10000, -10000, 10000, 0, 10000]
vertices = [[-4948.97959184, -6714.64019851],
            [-2448.97959184, -3141.43920596],
            [2448.97959184, 312.65508685],
            [6938.7755102, 5394.54094293],
            [4846.93877551, 6228.28784119],
            [2653.06122449, 3409.4292804],
            [-3520.40816327, -1434.24317618],
            [-6632.65306122, -6079.4044665]]
model = [PolygonalPrism(vertices, 1000, 4000, {'density': 1000})]
# and generate synthetic data from it
shape = (20, 20)
area = bounds[0:4]
x, y, z = gridder.regular(area, shape, z=-1)
noise = 0.1  # 0.1 mGal noise
gz = utils.contaminate(polyprism.gz(x, y, z, model), noise)

mpl.figure()
mpl.title("Gravity3D anomaly")
mpl.axis('scaled')
levels = mpl.contourf(y, x, gz, shape, 12)
mpl.colorbar()
mpl.xlabel('y (km)')
mpl.ylabel('x (km)')
mpl.m2km()
mpl.show()

# Inversion setup
mesh = PrismMesh(bounds, (25, 50, 50))
seeds = sow([[0, 0, 1500, {'density': 1000}]], mesh)
solver = Gravity3D(x, y, z, gz, mesh).config(
    'planting', seeds=seeds, compactness=0.05, threshold=0.0005).fit()
mesh.addprop('density', solver.estimate_)

# Plot the adjustment and the result
mpl.figure()
mpl.title("True: color | Predicted: contour")
mpl.axis('scaled')
levels = mpl.contourf(y, x, gz, shape, 12)
mpl.colorbar()
mpl.contour(y, x, solver.predicted(), shape, levels, color='k')
mpl.xlabel('y (km)')
mpl.ylabel('x (km)')
mpl.m2km()
mpl.show()
# Plot the result
myv.figure()
myv.polyprisms(model, 'density', opacity=0.6, linewidth=5)
myv.prisms(vremove(0, 'density', mesh), 'density')
myv.prisms(seeds, 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
         nlabels=3)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
