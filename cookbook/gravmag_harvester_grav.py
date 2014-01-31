"""
GravMag: 3D gravity inversion by planting anomalous densities using
``harvester`` (simple example)
"""
from fatiando import gridder, utils
from fatiando.gravmag import prism, harvester
from fatiando.mesher import Prism, PrismMesh, vremove
from fatiando.vis import mpl, myv

# Create a synthetic model
model = [Prism(250, 750, 250, 750, 200, 700, {'density':1000})]
# and generate synthetic data from it
shape = (25, 25)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
noise = 0.1 # 0.1 mGal noise
gz = utils.contaminate(prism.gz(xp, yp, zp, model), noise)
# plot the data
mpl.figure()
mpl.title("Synthetic gravity anomaly (mGal)")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, gz, shape, 12)
mpl.colorbar()
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
mpl.m2km()
mpl.show()

# Inversion setup
# Create a mesh
mesh = PrismMesh(bounds, (25, 25, 25))
# Wrap the data so that harvester can use it
data = [harvester.Gz(xp, yp, zp, gz)]
# Make the seed
seeds = harvester.sow([[500, 500, 450, {'density':1000}]], mesh)
# Run the inversioin
estimate, predicted = harvester.harvest(data, seeds, mesh,
    compactness=0.5, threshold=0.0005)

# Put the estimated density values in the mesh
mesh.addprop('density', estimate['density'])

# Plot the adjustment
mpl.figure()
mpl.title("True: color | Inversion: contour")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, gz, shape, 12)
mpl.colorbar()
mpl.contour(yp, xp, predicted[0], shape, levels, color='k')
mpl.xlabel('Horizontal coordinate y (km)')
mpl.ylabel('Horizontal coordinate x (km)')
mpl.m2km()
residuals = gz - predicted[0]
mpl.figure()
mpl.title('Residuals: mean=%g stddev=%g' % (residuals.mean(), residuals.std()))
mpl.hist(residuals, bins=10)
mpl.xlabel('Residuals (mGal)')
mpl.ylabel('# of')
mpl.show()
# Plot the result
myv.figure()
myv.prisms(model, 'density', style='wireframe')
myv.prisms(vremove(0, 'density', mesh), 'density')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
