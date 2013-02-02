"""
GravMag: 3D magnetic inversion by planting anomalous densities using
``harvester`` (simple example)
"""
from fatiando import logger, gridder, utils
from fatiando import gravmag as gm
from fatiando.mesher import Prism, PrismMesh, vremove
from fatiando.vis import mpl, myv

log = logger.get()
log.info(logger.header())

# Create a synthetic model
model = [Prism(250, 750, 250, 750, 200, 700, {'magnetization':1})]
# and generate synthetic data from it
shape = (51, 51)
bounds = [-500, 1500, -500, 1500, 0, 2000]
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
noise = 0.01 # 1 percent noise
inclination, declination = -20, -14
# Calculate the Total Field anomaly
tf = utils.contaminate(gm.prism.tf(xp, yp, zp, model, inclination, declination),
    noise, percent=True)
# plot the data
mpl.figure()
mpl.title("Synthetic total field anomaly (nT)")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, tf, shape, 30)
mpl.colorbar()
mpl.m2km()
mpl.show()

# Inversion setup
# Create a mesh
mesh = PrismMesh(bounds, (50, 50, 50))
# Wrap the data so that harvester can use it
data = [gm.harvester.TotalField(xp, yp, zp, tf, inclination, declination)]
# Make the seed
seeds = gm.harvester.sow([[500, 500, 450, {'magnetization':1}]], mesh)
# Run the inversioin
estimate, predicted = gm.harvester.harvest(data, seeds, mesh,
    compactness=0.5, threshold=0.0005)

# Put the estimated magnetization values in the mesh
mesh.addprop('magnetization', estimate['magnetization'])

# Plot the adjustment
mpl.figure()
mpl.title("True: color | Inversion: contour")
mpl.axis('scaled')
levels = mpl.contourf(yp, xp, tf, shape, 12)
mpl.colorbar()
mpl.contour(yp, xp, predicted[0], shape, levels, color='k')
mpl.m2km()
mpl.show()
# Plot the result
myv.figure()
myv.prisms(model, 'magnetization', style='wireframe')
myv.prisms(vremove(0, 'magnetization', mesh), 'magnetization')
myv.axes(myv.outline(bounds), ranges=[i*0.001 for i in bounds], fmt='%.1f',
    nlabels=6)
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.show()
