"""
Functions to read data from files and fetch datasets from the internet.
"""
from __future__ import absolute_import
from .surfer import load_surfer
from .utils import check_hash
from .hawaii_gravity import fetch_hawaii_gravity
from .icgem import load_icgem_gdf
from .image import from_image, SAMPLE_IMAGE, SAMPLE_IMAGE_SMALL
