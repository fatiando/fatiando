# Template parameter file for program harvester

# This is a comment. It will be ignored

# The name of the file with the data
data_file = 'data.txt'
# The extension can be anything you like (.txt, .xyz, .bla)
# The file should have the data in xyz format. That is:
# x1  y1  z1  height1  gz1  gxx1 ...
#   x2  y2  z2  height2  gz2  gxx2 ...
#   ...
#   xN  yN  zN  heightN  gzN  gxxN ...
# The coordinate system assumed is:
#    x -> North   y -> East   z -> Down
# Therefore, x is the Northing coordinate, y the Easting, and z the vertical
# coordinate. Optionally, height is the height of the topography (used to create
# a mesh that follows the topography). The order of the columns is not
# important.
# Remember: z is negative above the geoid but height is positive! gxx is the
#   North-North component, and so forth.
# Units: All coordinates and height are in meters! gz is in mGal and the tensor
#   components in Eotvos!

# Used to specify which columns of the data file should be used
#use_cols = None
use_cols = [0, 1, 2, 3, 4, 6, -1]  # -1 is the last. You can iterate backwards
# If it is None, will use all columns. If you want to leave out a column use
# something like (0 is the first column):
#   use_cols = [0, 1, 2, 3, 5]
# This way you can invert only some components without having to edit the data
# file.

# This is the column format. You should only give the name of the columns that
# will be used (i.e., if you removed some with use_cols, don't include them)!
# Possible names are: 'x', 'y', 'z', 'height', 'gz', 'gxx', 'gxy', 'gxz', 'gyy',
# 'gyz', 'gzz'
column_fmt = ['x', 'y', 'z', 'height', 'gz', 'gxy', 'gzz']

# Whether of not to invert the sign of z before running the inversion
inv_z = False
# Use inv_z = True if you want to turn it on. This is useful if your data set
# has z positive upward and you don't want to edit the data file.

# The boundaries of the mesh in meters:
#   [xmin, xmax, ymin, ymax]
mesh_bounds = None
# Set mesh_bounds = None and harvester will set the bounds as the limits of the
# data.

# The z coordinates of the top and bottom of the mesh in meters.
mesh_top = None  # Will place the top on the topography
mesh_bottom = 2000
# If you provided the 'height' column in the data file, then you can set
# mesh_top = None and harvester will place the top of the mesh on the topography

# The number of prisms in the x, y, and z directions
mesh_shape = (50, 50, 25)

# The file with the seeds.
seed_file = 'seeds.txt'
# The seed file is in JSON format and should be like this:
#
# [
#   [x1, y1, z1, {"density":dens1}],
#   [x2, y2, z2, {"density":dens2, "magnetization":mag2}],
#   [x3, y3, z3, {"magnetization":mag3, "inclination":inc3,
#                 "declination":dec3}],
#   ...
# ]
#
# x, y, z are the coordinates of the seed and the dict (``{'density':2670}``)
# are its physical properties.
# WARNING: Must use ", not ', in the physical property names!#
# Each seed can have different kinds of physical properties. If inclination
# and declination are not given, will use the inc and dec of the inducing
# field (i.e., no remanent magnetization).
# Again, white space and newlines don't matter and the file extension can be
# anything.

# The value of the regularizing parameter. Must be >= 0.
regul = 1
# The regularizing parameter controls how strongly the compactness
# regularization is imposed. The higher this value, the more it is imposed.
# In practice, there is a limit to how much compactness you'll get.

# The threshold value for how small a change in the data-misfit is accepted
delta = 0.0005
# This controls how much the solution is allowed to grow. If it's too big, the
# seeds won't grow.

# Output files for the estimated physical property distribution and the mesh

# Name of the output file in Python's pickle format (so that it can be loaded
# from other scripts). Saves the fatiando.msh.ddd.PrismMesh instance with its
# physical properties set to the estimate.
pickle_file = None

# Name of the output files in the format accepted by the UBC-GIF software
# Meshtools <http://www.eos.ubc.ca/research/ubcgif>.
mesh_file = 'result.msh'
density_file = 'result.den'

# Name of the file where the predicted data (modeled) will be saved.
pred_file = 'predicted.txt'
# The format will be the same as the input data file. Again, the file extension
# can be anything.

