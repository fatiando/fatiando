r"""
Finite difference solution of the 2D wave equation for isotropic media.

* :func:`~fatiando.seismic.wavefd.elastic_psv`: Simulates the coupled P and SV
  elastic waves using the Parsimonious Staggered Grid method of Luo and
  Schuster (1990)
* :func:`~fatiando.seismic.wavefd.elastic_sh`: Simulates SH elastic waves using
  the Equivalent Staggered Grid method of Di Bartolo et al. (2012)
* :func:`~fatiando.seismic.wavefd.scalar`: Simulates scalar waves using simple
  explicit finite differences scheme

**Sources**

* :class:`~fatiando.seismic.wavefd.MexHatSource`: Mexican hat wavelet source
* :class:`~fatiando.seismic.wavefd.SinSqrSource`: Sine squared source
* :class:`~fatiando.seismic.wavefd.GaussSource`: Gauss derivative source
* :func:`~fatiando.seismic.wavefd.blast_source`: A source blasting in all
  directions

**Auxiliary functions**

* :func:`~fatiando.seismic.wavefd.lame_lamb`: Calculate the lambda Lame
  parameter
* :func:`~fatiando.seismic.wavefd.lame_mu`: Calculate the mu Lame parameter
* :func:`~fatiando.seismic.wavefd.xz2ps`: Convert x and z displacements to
  representations of P and S waves
* :func:`~fatiando.seismic.wavefd.maxdt`: Calculate the maximum time step for
  elastic wave simulations
* :func:`~fatiando.seismic.wavefd.scalar_maxdt`: Calculate the maximum time
  step for a scalar wave simulation

**Theory**

A good place to start is the equation of motion for elastic isotropic materials

.. math::

    \partial_j \tau_{ij} - \rho \partial_t^2 u_i = -f_i

where :math:`\tau_{ij}` is the stress tensor, :math:`\rho` the density,
:math:`u_i` the displacement (particle motion) and :math:`f_i` is the source
term.
But what I'm interested in modeling are the displacements in x, y and z.
They are what is recorded by the seismometers.
Luckily, I can use Hooke's law to write the stress tensor as a function of the
displacements

.. math::

    \tau_{ij} = \lambda\delta_{ij}\partial_k u_k +
    \mu(\partial_i u_j + \partial_j u_i)

where :math:`\lambda` and :math:`\mu` are the Lame parameters and
:math:`\delta_{ij}` is the Kronecker delta.
Just as a reminder, in tensor notation, :math:`\partial_k u_k` is the divergent
of :math:`\mathbf{u}`. Free indices (not :math:`i,j`) represent a summation.

In a 2D medium, there is no variation in one of the directions.
So I'll choose the y direction for this purpose, so all y-derivatives cancel
out.
Looking at the second component of the equation of motion

.. math::

    \partial_x\tau_{xy} + \underbrace{\partial_y\tau_{yy}}_{0} +
    \partial_z\tau_{yz}  -  \rho \partial_t^2 u_y = -f_y

Substituting the stress formula in the above equation yields

.. math::

    \partial_x\mu\partial_x u_y  + \partial_z\mu\partial_z u_y
    - \rho\partial_t^2 u_y = -f_y

which is the wave equation for horizontally polarized S waves, i.e. SH waves.
This equation is solved here using the Equivalent Staggered Grid (ESG) method
of Di Bartolo et al. (2012).
This method was developed for acoustic (pressure) waves but can be applied
without modification to SH waves.

Canceling the y derivatives in the first and third components of the equation
of motion

.. math::

    \partial_x\tau_{xx} + \partial_z\tau_{xz} - \rho \partial_t^2 u_x = -f_x

.. math::

    \partial_x\tau_{xz} + \partial_z\tau_{zz} - \rho \partial_t^2 u_z = -f_z

And the corresponding stress components are

.. math::

    \tau_{xx} = (\lambda + 2\mu)\partial_x u_x + \lambda\partial_z u_z

.. math::

    \tau_{zz} = (\lambda + 2\mu)\partial_z u_z + \lambda\partial_x u_x

.. math::

    \tau_{xz} = \mu( \partial_x u_z + \partial_z u_x)

This means that the displacements in x and z are coupled and must be solved
together.
This equation describes the motion of pressure (P) and vertically polarized S
waves (SV).
The method used here to solve these equations is the Parsimonious Staggered
Grid (PSG) of Luo and Schuster (1990).


**References**

Di Bartolo, L., C. Dors, and W. J. Mansur (2012), A new family of
finite-difference schemes to solve the heterogeneous acoustic wave equation,
Geophysics, 77(5), T187-T199, doi:10.1190/geo2011-0345.1.

Luo, Y., and G. Schuster (1990), Parsimonious staggered grid
finite-differencing of the wave equation, Geophysical Research Letters, 17(2),
155-158, doi:10.1029/GL017i002p00155.

----

"""
