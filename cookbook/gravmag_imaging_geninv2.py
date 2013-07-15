"""
GravMag: 3D imaging using the Generalized Inverse method on synthetic gravity
data (more complex model + noisy data)
"""
from fatiando import gridder, mesher, gravmag, utils
from fatiando.vis import mpl, myv

# Make some synthetic gravity data from a simple prism model
prisms = [mesher.Prism(-4000,-1000,-4000,-2000,2000,5000,{'density':800}),
          mesher.Prism(-1000,1000,-1000,1000,1000,6000,{'density':-800}),
          mesher.Prism(2000,4000,3000,4000,0,4000,{'density':600})]
shape = (25, 25)
xp, yp, zp = gridder.regular((-5000, 5000, -5000, 5000), shape, z=-10)
gz = utils.contaminate(gravmag.prism.gz(xp, yp, zp, prisms), 0.1)

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
mesh = gravmag.imaging.geninv(xp, yp, zp, gz, shape,
    0, 10000, 25)

# Plot the results
myv.figure()
myv.prisms(prisms, 'density', style='wireframe')
myv.prisms(mesh, 'density', edges=False, linewidth=5)
axes = myv.axes(myv.outline())
myv.wall_bottom(axes.axes.bounds)
myv.wall_north(axes.axes.bounds)
myv.show()
