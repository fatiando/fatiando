"""
GravMag: 3D forward modeling of total-field magnetic anomaly using rectangular
prisms (model with induced and remanent magnetization)
"""
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism
from fatiando.vis import mpl, myv

# The regional field
inc, dec = 30, -15
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    mesher.Prism(-4000, -3000, -4000, -3000, 0, 2000,
                 {'magnetization': utils.ang2vec(1, inc, dec)}),
    mesher.Prism(-1000, 1000, -1000, 1000, 0, 2000,
                 {'magnetization': utils.ang2vec(1, inc, dec)}),
    # This prism will have magnetization in a different direction
    mesher.Prism(2000, 4000, 3000, 4000, 0, 2000,
                 {'magnetization': utils.ang2vec(3, -10, 45)})]
# Create a regular grid at 100m height
shape = (200, 200)
area = bounds[:4]
xp, yp, zp = gridder.regular(area, shape, z=-500)
# Calculate the anomaly for a given regional field
tf = prism.tf(xp, yp, zp, model, inc, dec)
# Plot
mpl.figure()
mpl.title("Total-field anomaly (nT)")
mpl.axis('scaled')
mpl.contourf(yp, xp, tf, shape, 15)
mpl.colorbar()
mpl.xlabel('East y (km)')
mpl.ylabel('North x (km)')
mpl.m2km()
mpl.show()
# Show the prisms
myv.figure()
myv.prisms(model, 'magnetization')
myv.axes(myv.outline(bounds), ranges=[i * 0.001 for i in bounds])
myv.wall_north(bounds)
myv.wall_bottom(bounds)
myv.show()
