"""
GravMag: 3D imaging using the migration method on synthetic gravity data
(more complex model + noisy data)
"""
from fatiando import gridder, mesher, utils
from fatiando.gravmag import prism, imaging
from fatiando.vis import mpl, myv

# Make some synthetic gravity data from a simple prism model
model = [mesher.Prism(-4000, 0, -4000, -2000, 2000, 5000, {'density': 1200}),
         mesher.Prism(-1000, 1000, -1000, 1000, 1000, 7000, {'density': -800}),
         mesher.Prism(2000, 4000, 3000, 4000, 0, 2000, {'density': 600})]
# Calculate on a scatter of points to show that migration doesn't need gridded
# data
xp, yp, zp = gridder.scatter((-6000, 6000, -6000, 6000), 1000, z=-10)
gz = utils.contaminate(prism.gz(xp, yp, zp, model), 0.1)

# Plot the data
shape = (50, 50)
mpl.figure()
mpl.axis('scaled')
mpl.contourf(yp, xp, gz, shape, 30, interp=True)
mpl.colorbar()
mpl.plot(yp, xp, '.k')
mpl.xlabel('East (km)')
mpl.ylabel('North (km)')
mpl.m2km()
mpl.show()

mesh = imaging.migrate(xp, yp, zp, gz, 0, 10000, (30, 30, 30), power=0.8)

# Plot the results
myv.figure()
myv.prisms(model, 'density', style='wireframe', linewidth=2)
myv.prisms(mesh, 'density', edges=False)
axes = myv.axes(myv.outline())
myv.wall_bottom(axes.axes.bounds)
myv.wall_north(axes.axes.bounds)
myv.show()
