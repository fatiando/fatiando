"""
GravMag: 3D imaging using the Generalized Inverse method on synthetic gravity
data (simple model)
"""
from fatiando import gridder, mesher
from fatiando.gravmag import prism, imaging
from fatiando.vis import mpl, myv

# Make some synthetic gravity data from a simple prism model
model = [mesher.Prism(-1000,1000,-3000,3000,0,5000,{'density':1000})]
shape = (25, 25)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-10)
gz = prism.gz(xp, yp, zp, model)

# Plot the data
mpl.figure()
mpl.axis('scaled')
mpl.contourf(yp, xp, gz, shape, 30)
mpl.colorbar()
mpl.xlabel('East (km)')
mpl.ylabel('North (km)')
mpl.m2km()
mpl.show()

# Run the Generalized Inverse
mesh = imaging.geninv(xp, yp, zp, gz, shape, 0, 10000, 25)

# Plot the results
myv.figure()
myv.prisms(model, 'density', style='wireframe', linewidth=5)
myv.prisms(mesh, 'density', edges=False)
axes = myv.axes(myv.outline())
myv.wall_bottom(axes.axes.bounds)
myv.wall_north(axes.axes.bounds)
myv.show()
