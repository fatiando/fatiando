import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from nose.tools import assert_raises

from fatiando.seismic import conv


def test_density_matrix_input
#rho default com rho matrix=1, result the same
    #model parameters
    n_samples, n_traces = [600, 500]
    rock_grid = 1500.*np.ones((n_samples, n_traces))
    rock_grid[300:,:] = 2500.
    #synthetic calculation for rho as int
    [vel_l_int, rho_l_int] = conv.depth_2_time(n_samples, n_traces,
                                                rock_grid, dt=2.e-3)
    synt_int = conv.seismic_convolutional_model(n_traces, vel_l_int,
                                                30., conv.rickerwave)
    #synthetic calculation for rho as matrix
    [vel_l_mat,rho_l_mat] = conv.depth_2_time(n_samples, 
           n_traces, rock_grid, dt=2.e-3,rho=1.*np.ones((n_samples, n_traces)))
    synt_mat = conv.seismic_convolutional_model(n_traces, vel_l_mat,
                                           30., conv.rickerwave, rho=rho_l_mat)
    assert_array_almost_equal(synt_int, synt_mat, 9)

