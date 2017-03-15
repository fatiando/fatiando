"""
Generate and operate on various kinds of meshes and geometric elements
"""
from __future__ import absolute_import
from .geometry import Polygon, Square, Prism, PolygonalPrism, Sphere, Tesseroid
from .mesh2d import SquareMesh
from .mesh3d import PrismMesh  # , PointGrid, TesseroidMesh
from .relief import PrismRelief
