"""
Wrappers for calls to Mayavi2's `mlab` module for plotting
:mod:`~fatiando.mesher.ddd` objects and automating common tasks.

**Objects**

* :func:`~fatiando.vis.vtk.prisms`

**Helpers**

* :func:`~fatiando.vis.vtk.figure`
* :func:`~fatiando.vis.vtk.add_outline`
* :func:`~fatiando.vis.vtk.add_axes`
* :func:`~fatiando.vis.vtk.wall_north`
* :func:`~fatiando.vis.vtk.wall_south`
* :func:`~fatiando.vis.vtk.wall_east`
* :func:`~fatiando.vis.vtk.wall_west`
* :func:`~fatiando.vis.vtk.wall_top`
* :func:`~fatiando.vis.vtk.wall_bottom`

----
   
"""

import numpy

from fatiando import logger


log = logger.dummy('fatiando.vis.vtk')
        
# Do lazy imports of mlab and tvtk to avoid the slow imports when I don't need
# 3D plotting
mlab = None
tvtk = None

def _lazy_import_mlab():
    """
    Do the lazy import of mlab
    """
    global mlab
    # For campatibility with versions of Mayavi2 < 4
    if mlab is None:
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

def _lazy_import_tvtk():
    """
    Do the lazy import of tvtk
    """
    global tvtk
    # For campatibility with versions of Mayavi2 < 4
    if tvtk is None:
        try:
            from tvtk.api import tvtk
        except ImportError:
            from enthought.tvtk.api import tvtk

def prisms(prisms, prop=None, style='surface', opacity=1, edges=True, vmin=None,
    vmax=None, cmap='blue-red'):
    """
    Plot a list of 3D right rectangular prisms using Mayavi2.

    Will not plot a value None in *prisms*

    Parameters:
    
    * prisms : list
        Prisms (see :class:`~fatiando.mesher.ddd.Prism`)
    * prop : str or None
        The physical property of the prisms to use as the color scale. If a
        prism doesn't have *prop*, or if it is None, then it will not be plotted
    * style : str
        Either ``'surface'`` for solid prisms or ``'wireframe'`` for just the
        contour
    * opacity : float
        Decimal percentage of opacity
    * edges : True or False
        Wether or not to display the edges of the prisms in black lines. Will
        ignore this if ``style='wireframe'``
    * vmin, vmax : float
        Min and max values for the color scale. If *None* will default to
        the min and max of *prop* in the prisms.
    * cmap : Mayavi colormap
        Color map to use. See the 'Colors and Legends' menu on the Mayavi2 GUI
        for valid color maps.

    Returns:
    
    * surface
        the last element on the pipeline

    """
    if style not in ['surface', 'wireframe']:
        raise ValueError, "Invalid style '%s'" % (style)
    if opacity > 1. or opacity < 0:
        msg = "Invalid opacity %g. Must be in range [1,0]" % (opacity)
        raise ValueError, msg

    # mlab and tvtk are really slow to import
    _lazy_import_mlab()
    _lazy_import_tvtk()

    if prop is None:
        label = 'scalar'
    else:
        label = prop
    # VTK parameters
    points = []
    cells = []
    offsets = []
    offset = 0
    mesh_size = 0
    celldata = []
    # To mark what index in the points the cell starts
    start = 0
    for prism in prisms:
        if prism is None or (prop is not None and prop not in prism.props):
            continue
        x1, x2, y1, y2, z1, z2 = prism.get_bounds()
        if prop is None:
            scalar = 0.
        else:
            scalar = prism.props[prop]
        points.extend([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                       [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])
        cells.append(8)
        cells.extend([i for i in xrange(start, start + 8)])
        start += 8
        offsets.append(offset)
        offset += 9
        celldata.append(scalar)
        mesh_size += 1
    cell_array = tvtk.CellArray()
    cell_array.set_cells(mesh_size, numpy.array(cells))
    cell_types = numpy.array([12]*mesh_size, 'i')
    vtkmesh = tvtk.UnstructuredGrid(points=numpy.array(points, 'f'))
    vtkmesh.set_cells(cell_types, numpy.array(offsets, 'i'), cell_array)
    vtkmesh.cell_data.scalars = numpy.array(celldata)
    vtkmesh.cell_data.scalars.name = label
    dataset = mlab.pipeline.add_dataset(vtkmesh)
    if vmin is None:
        vmin = min(vtkmesh.cell_data.scalars)
    if vmax is None:
        vmax = max(vtkmesh.cell_data.scalars)
    surf = mlab.pipeline.surface(dataset, vmax=vmax, vmin=vmin, colormap=cmap)
    if style == 'wireframe':
        surf.actor.property.representation = 'wireframe'
    if style == 'surface':
        surf.actor.property.representation = 'surface'
        if edges:
            surf.actor.property.edge_visibility = 1
    surf.actor.property.opacity = opacity
    surf.actor.property.backface_culling = 1
    return surf

def figure(size=None):
    """
    Create a default figure in Mayavi with white background and z pointing down

    Parameters:

    * size : tuple = (dx, dy)
        The size of the figure. If ``None`` will use the default size.

    Return:
    
    * fig : Mayavi figure object
        The figure

    """
    _lazy_import_mlab()
    if size is None:
        fig = mlab.figure(bgcolor=(1, 1, 1))
    else:
        fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    fig.scene.camera.view_up = numpy.array([0., 0., -1.])
    fig.scene.camera.elevation(60.)
    fig.scene.camera.azimuth(180.)
    return fig

def add_outline(extent=None, color=(0,0,0), width=2):
    """
    Create a default outline in Mayavi2.

    Parameters:
    
    * extent : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Default if the objects extent.
    * color : tuple = (r, g, b)
        RGB of the color of the axes and text
    * width : float
        Line width

    Returns:
    
    * outline : Mayavi outline instace
        The outline in the pipeline

    """
    _lazy_import_mlab()
    outline = mlab.outline(color=color, line_width=width)
    if extent is not None:
        outline.bounds = extent
    return outline

def add_axes(plot, nlabels=5, extent=None, ranges=None, color=(0,0,0),
             width=2, fmt="%-#.2f"):
    """
    Add an Axes module to a Mayavi2 plot or dataset.

    Parameters:
    
    * plot
        Either the plot (as returned by one of the plotting functions of this
        module) or a TVTK dataset.
    * nlabels : int
        Number of labels on the axes
    * extent : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Default if the objects extent.
    * ranges : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        What will be display in the axes labels. Default is *extent*
    * color : tuple = (r, g, b)
        RGB of the color of the axes and text
    * width : float
        Line width
    * fmt : str
        Label number format

    Returns:
    
    * axes : Mayavi axes instace
        The axes object in the pipeline

    """
    _lazy_import_mlab()
    a = mlab.axes(plot, nb_labels=nlabels, color=color)
    a.label_text_property.color = color
    a.title_text_property.color = color
    if extent is not None:
        a.axes.bounds = extent
    if ranges is not None:
        a.axes.ranges = ranges
        a.axes.use_ranges = True
    a.property.line_width = width
    a.axes.label_format = fmt
    a.axes.x_label, a.axes.y_label, a.axes.z_label = "N", "E", "Z"
    return a

def wall_north(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the North side.

    .. note:: Remember that x->North, y->East and z->Down

    Parameters:
    
    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        The extent of the region where the wall is placed
    * color : tuple = (r, g, b)
        RGB of the color of the wall
    * opacity : float
        Decimal percentage of opacity

    .. tip:: You can use :func:`~fatiando.vis.vtk.add_axes` to create and
        `axes` variable and get the bounds as ``axes.axes.bounds``

    """
    s, n, w, e, t, b = bounds
    _wall([n, n, w, e, b, t], color, opacity)

def wall_south(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the South side.

    .. note:: Remember that x->North, y->East and z->Down

    Parameters:
    
    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        The extent of the region where the wall is placed
    * color : tuple = (r, g, b)
        RGB of the color of the wall
    * opacity : float
        Decimal percentage of opacity

    .. tip:: You can use :func:`~fatiando.vis.vtk.add_axes` to create and
        `axes` variable and get the bounds as ``axes.axes.bounds``

    """
    s, n, w, e, t, b = bounds
    _wall([s, s, w, e, b, t], color, opacity)

def wall_east(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the East side.

    .. note:: Remember that x->North, y->East and z->Down

    Parameters:
    
    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        The extent of the region where the wall is placed
    * color : tuple = (r, g, b)
        RGB of the color of the wall
    * opacity : float
        Decimal percentage of opacity

    .. tip:: You can use :func:`~fatiando.vis.vtk.add_axes` to create and
        `axes` variable and get the bounds as ``axes.axes.bounds``

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, e, e, b, t], color, opacity)

def wall_west(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the West side.

    .. note:: Remember that x->North, y->East and z->Down

    Parameters:
    
    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        The extent of the region where the wall is placed
    * color : tuple = (r, g, b)
        RGB of the color of the wall
    * opacity : float
        Decimal percentage of opacity

    .. tip:: You can use :func:`~fatiando.vis.vtk.add_axes` to create and
        `axes` variable and get the bounds as ``axes.axes.bounds``

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, w, w, b, t], color, opacity)

def wall_top(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the Top side.

    .. note:: Remember that x->North, y->East and z->Down

    Parameters:
    
    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        The extent of the region where the wall is placed
    * color : tuple = (r, g, b)
        RGB of the color of the wall
    * opacity : float
        Decimal percentage of opacity

    .. tip:: You can use :func:`~fatiando.vis.vtk.add_axes` to create and
        `axes` variable and get the bounds as ``axes.axes.bounds``

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, w, e, t, t], color, opacity)

def wall_bottom(bounds, color=(0,0,0), opacity=0.1):
    """
    Draw a 3D wall in Mayavi2 on the Bottom side.

    .. note:: Remember that x->North, y->East and z->Down

    Parameters:
    
    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        The extent of the region where the wall is placed
    * color : tuple = (r, g, b)
        RGB of the color of the wall
    * opacity : float
        Decimal percentage of opacity

    .. tip:: You can use :func:`~fatiando.vis.vtk.add_axes` to create and
        `axes` variable and get the bounds as ``axes.axes.bounds``

    """
    s, n, w, e, t, b = bounds
    _wall([s, n, w, e, b, b], color, opacity)

def _wall(bounds, color, opacity):
    """Generate a 3D wall in Mayavi"""
    _lazy_import_mlab()
    p = mlab.pipeline.builtin_surface()
    p.source = 'outline'
    p.data_source.bounds = bounds
    p.data_source.generate_faces = 1
    su = mlab.pipeline.surface(p)
    su.actor.property.color = color
    su.actor.property.opacity = opacity
