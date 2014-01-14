"""
GravMag: Classic 3D Euler deconvolution of magnetic data using a
moving window
"""
from fatiando.mesher import Prism
from fatiando import gridder, utils
from fatiando.gravmag import prism, fourier
from fatiando.gravmag.euler import Classic, MovingWindow
from fatiando.vis import mpl, myv

# Make a model
bounds = [-5000, 5000, -5000, 5000, 0, 5000]
model = [
    Prism(-1500, -500, -1500, -500, 500, 1500, {'density':1000}),
    Prism(500, 1500, 1000, 2000, 500, 1500, {'density':1000})]
# Generate some data from the model
shape = (100, 100)
area = bounds[0:4]
xp, yp, zp = gridder.regular(area, shape, z=-1)
# Add a constant baselevel
baselevel = 10
# Convert the data from mGal to SI because Euler and FFT derivation require
# data in SI
gz = utils.mgal2si(prism.gz(xp, yp, zp, model)) + baselevel
xderiv = fourier.derivx(xp, yp, gz, shape)
yderiv = fourier.derivy(xp, yp, gz, shape)
zderiv = fourier.derivz(xp, yp, gz, shape)

mpl.figure()
titles = ['Gravity anomaly', 'x derivative', 'y derivative', 'z derivative']
for i, f in enumerate([gz, xderiv, yderiv, zderiv]):
    mpl.subplot(2, 2, i + 1)
    mpl.title(titles[i])
    mpl.axis('scaled')
    mpl.contourf(yp, xp, f, shape, 50)
    mpl.colorbar()
    mpl.m2km()
mpl.show()

# Run the euler deconvolution on moving windows to produce a set of solutions
euler = Classic(xp, yp, zp, gz, xderiv, yderiv, zderiv, 2)
solver = MovingWindow(euler, windows=(10, 10), size=(2000, 2000)).fit()

mpl.figure()
mpl.axis('scaled')
mpl.title('Moving window centers')
mpl.contourf(yp, xp, gz, shape, 50)
mpl.points(solver.window_centers)
mpl.show()

myv.figure()
myv.points(solver.estimate_, size=100.)
myv.prisms(model, opacity=0.5)
axes = myv.axes(myv.outline(bounds), ranges=[b*0.001 for b in bounds])
myv.wall_bottom(bounds)
myv.wall_north(bounds)
myv.title('Euler solutions')
myv.show()
