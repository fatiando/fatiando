"""
Create and operate on data grids, scatters, and profiles.
"""
from __future__ import absolute_import
from .slicing import inside, cut
from .interpolation import interp, interp_at, profile
from .padding import pad_array, unpad_array, pad_coords
from .point_generation import regular, scatter, circular_scatter
from .utils import spacing
