"""
Pad the edges of grids using various methods
=============================================

Sometimes it is useful to add some padding points to the edges of grids, for
example during FFT-based processing to avoid edge effects.
Function :func:`fatiando.gridder.pad_array` does this using various padding
methods.
Functions
:func:`fatiando.gridder.unpad_array` (to remove padding) and
:func:`fatiando.gridder.pad_coords` (to created padded coordinate arrays)
offer support for common operations done while padding.
"""
import matplotlib.pyplot as plt
import numpy as np
from fatiando import gridder

# Generate some synthetic data
area = (-100, 100, -60, 60)
shape = (101, 172)
# The padding functions need data to be on a regular grid and represented by a
# 2D numpy array. So I'll convert the outputs to 2D.
x, y = gridder.regular(area, shape)
x = x.reshape(shape)
y = y.reshape(shape)
data = np.sin(0.1*x)*np.cos(0.09*y) + 0.001*(x**2 + y**2)

# Pad arrays with all the padding options and make a single figure with all of
# them.
fig, axes = plt.subplots(2, 4, figsize=(10, 6), sharex=True, sharey=True)

ax = axes[0, 0]
ax.set_title('Original')
# Keep all plots on the same color scale of the original data
vmin, vmax = data.min(), data.max()
ax.pcolormesh(y, x, data, cmap='RdBu_r', vmin=vmin, vmax=vmax)

padtypes = ['0', 'mean', 'edge', 'lintaper', 'reflection', 'oddreflection',
            'oddreflectiontaper']
for padtype, ax in zip(padtypes, axes.ravel()[1:]):
    padded_data, nps = gridder.pad_array(data, padtype=padtype)
    # Get coordinate vectors
    pad_x, pad_y = gridder.pad_coords([x, y], shape, nps)
    padshape = padded_data.shape
    ax.set_title(padtype)
    ax.pcolormesh(pad_y.reshape(padshape), pad_x.reshape(padshape),
                  padded_data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xlim(pad_y.min(), pad_y.max())
    ax.set_ylim(pad_x.min(), pad_x.max())
plt.tight_layout(w_pad=0)
plt.show()
