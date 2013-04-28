GOALS FOR 0.2:

* Make lazy imports in myv decorators
* Get rid of logging. Replace for decorator that records only debuging func
  name and args
* make utils.timed decorator to print timing information to a stream
* Write the Advanced usage docs
* Write the Getting started docs
* Make a VTK plotter for tesseroids
* Optimize PrismMesh.dump (using Cython?)
* Add titles, figures and better description to recipe docstrings
* Make plot for spheres in 3D
* Potential field compact inversion in 2D
* Finish numexpr and Cython modules for polyprism
* Make fatiando.io for easy pickling, json, grid IO, etc
* Make a msh.ddd.Point3d object and make vis.vtk.points3d plot it with physical
  properties
* logger.get should receive a list of streams and add handlers to all
* Make logger get the level from string instead of having to import logging
* Add option for scalar visibility in vis.vtk.prisms
* Add data custom data weights to harvester data modules
* Make Cython extension for sparse Jacobian computation in seis.srtomo
* Make parallel building of Jacobians in seis.srtomo
* Get rid of use_sparse functions, instead, use sparse=True keyword arguments
* Fix recipe epicenter_regularized.py
* Optimize equality constraints to use less memory
* Remove MinimumDistance from epic2d
* Make unit tests to test Cython prism against numpy prism
* Make unit test for talwani comparing with prism
* Make unit test for polyprism comparing with prism
* Finish the tutorial for harvester
* Write the overview section in docs
* Update formula in the docstring of seis.wavefd (the code is fourth order)
* Instructions for getting the latest dev from bitbucket
* Instructions for installing .zip and tar archives with pip, not setup.py
* Instructions for building Cython modules on Windows

URGENT:

* Put 'not None' in all Cython function arguments
* gravmag.tesseroid: move sin and cos of latitudes out of kernels

BUGS:

* title3d needs be called after a call to a data plotter (e.g., prisms)
* BUG in talwani.c: wrong result when zv==zvp1 (and a little wrong when xv==0)

GOALS FOR 0.3:

* Write Developers Guide:
    * Coding style
    * Using version control
    * Getting the source
    * Building
    * Sending patches
* Write tutorials with a larger examples than in the cookbook

IDEAS:

* Try multithreading and multiprocessing sensitivity building
* Importer functions to fetch DEMs, gravity data form IGBE, etc
* Make utils.clock function that runs a function, logs the time it takes and
  returns what the func would return
* Get rid of the DataModule class. Provide a specification of it in the docs on
  fatiando.inversion
* Add decorator to logging that logs a functions name and parameters in debug
  mode
* Consider making the parameter vector a dictionary: one array for each prop.
  this way the datamodules only operate on their props
* Make an automatic fetcher of bibliographic references from scripts
* Store provenance of results in image files:
  http://galacticusblog.blogspot.com/2012/01/reproducibility-of-galacticus-modesl.html
* Use diagonal derivatives in Smoothness

TO-IMPLEMENT:

* Implement PML absorbing boundary conditions in seis.wavefd. Using a large grid
  and exponential decay factor instead
* Implement potential forward modeling using FFT
* Parkers algorithm for potential fields
* Fully integrate Radial3D inversion
* Make a PolyprismStack that creates PolygonalPrisms staked on top of each other
* Analytical upward continuation for tensor components
* Spherical harmonics for potential fields
* Spherical harmonic equivalent of Parkers algorithm
* Sparsity regularization
* McKenzie model
* VBA model for oceanic lithosphere
* ifat script that loads IPython with numpy and fatiando
* potential.basin2d inversion using prisms
* Interactive 2D potential field inversion (Silva and Barbosa, 2006)
* Regional removal in potential fields
* potential.sphere with the effect of a sphere
* Finish implementing potential.polyprism
* gridder.rotate to rotate grids
* gridder.load/save to wrap numpy.loadtxt/savetxt (optional support for grid formats)
* gridder.stream to return an interator that reads one grid point at a time
* gridder.profile to extract (interpolate) a profile from a grid
* mesher.ddd.prism2vtk to convert prisms to VTK and dump it to a file
* inversion.linear.undet underdetermined solver for linear problems
* Refactor ui.picker.draw_polygon to draw many polygons
* utils.erange A range generator function with exponentially increasing intervals
* Make a PrismMesh.get_index(i) method that converts index i in raveled array to
  i, j, k 3D index
* potential.terrain for terrain corrections (automatically find the best density)
