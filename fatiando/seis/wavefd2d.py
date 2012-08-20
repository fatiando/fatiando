r"""
Finite difference solution of the 2D wave equation for isotropic media.

Simulates both elastic and acoustic waves:

* :func:`~fatiando.seis.wavefd2d.elastic_psv`: Simulates the coupled P and SV
  elastic waves
* :func:`~fatiando.seis.wavefd2d.elastic_sh`: Simulates SH elastic waves


**Theory**

We start with the wave equation for elastic isotropic media

.. math::

    (\lambda + \mu)\nabla(\nabla\cdot\mathbf{u})
    +
    \mu\nabla^2\mathbf{u}
    - \rho \partial_t^2 \mathbf{u} = - \mathbf{f}

where :math:`\mathbf{u} = (u_x, u_y, y_z)` is the particle movement vector,
:math:`\rho` is the density,
:math:`\lambda` and :math:`\mu` are the Lame constants,
and :math:`\mathbf{f}` is the source vector.

In the 2D approximation, we assume all derivatives in the y direction
are zero and consider only x and z coordinates
(though :math:`u_y` remains).
The three equations in the vector equation above can be separated into
two groups:

.. math::

    \mu\left(\partial_x^2 u_y + \partial_z^2 u_y\right)
    - \rho \partial_t^2 u_y = f_y

and

.. math::

    (\lambda + 2\mu)\partial_x^2 u_x + \mu\partial_z^2 u_x
    + (\lambda + \mu)\partial_x\partial_z u_z
    - \rho \partial_t^2 u_x = f_x

.. math::

    (\lambda + 2\mu)\partial_z^2 u_z + \mu\partial_x^2 u_z
    + (\lambda + \mu)\partial_x\partial_z u_x
    - \rho \partial_t^2 u_z = f_z

The first equation depends only on :math:`u_y` and represents SH waves.
The other two depend on :math:`u_x` and :math:`u_z` and are coupled.
They represent P and SV waves.

The explicit finite difference solution for the SH waves is:

.. math::
   :nowrap:

    \begin{align*}
        u_y[i,j]_{t+1} =& 2u_y[i,j]_{t} - u_y[i,j]_{t-1}
        + \frac{\Delta t^2}{\rho[i,j]}
        \left[
            -f_y[i,j]_t +
            \mu[i,j]
                \left(
                    \frac{u_y[i+1,j]_{t} - 2u_y[i,j]_{t} + u_y[i-1,j]_{t}}{
                        \Delta x^2}
                \right.
        \right.
        \\[0.3cm] &
        \left.
            \left.
                +
                \frac{u_y[i,j+1]_{t} - 2u_y[i,j]_{t} + u_y[i,j-1]_{t}}{\Delta z^2}
            \right)
        \right]
    \end{align*}


where :math:`[i,j]_t` is the quantity at the grid node i,j at a
time t.

The solution for P and SV waves is:

.. math::
   :nowrap:

    \begin{align*}
    u_x[i,j]_{t+1} =& 2u_x[i,j]_{t} - u_x[i,j]_{t-1}
    \\&
    + \frac{\Delta t^2}{\rho[i,j]}
    \left\lbrace
        -f_x[i,j]_{t} +
        (\lambda[i,j] + 2\mu[i,j])
        \left(
            \frac{u_x[i+1,j]_{t} - 2u_x[i,j]_{t} + u_x[i-1,j]_{t}}{
                \Delta x^2}
        \right)
    \right.
    \\[0.3cm]&
        +
        \mu[i,j]
        \left(
            \frac{u_x[i,j+1]_{t} - 2u_x[i,j]_{t} + u_x[i,j-1]_{t}}{
                \Delta z^2}
        \right)
    \\[0.3cm]&
    \left.
        +
        (\lambda[i,j] + \mu[i,j])
        \left(
            \frac{u_z[i,j]_{t} - u_z[i-1,j]_{t} - u_z[i,j-1]_{t} +
            u_z[i-1,j-1]_{t}
            }{\Delta x\Delta z}
        \right)
    \right\rbrace
    \end{align*}

.. math::
   :nowrap:

    \begin{align*}
    u_z[i,j]_{t+1} =& 2u_z[i,j]_{t} - u_z[i,j]_{t-1}
    \\&
    + \frac{\Delta t^2}{\rho[i,j]}
    \left\lbrace
        -f_z[i,j]_{t} +
        (\lambda[i,j] + 2\mu[i,j])
        \left(
            \frac{u_z[i,j+1]_{t} - 2u_z[i,j]_{t} + u_z[i,j-1]_{t}}{
                \Delta z^2}
        \right)
    \right.
    \\[0.3cm]&
        +
        \mu[i,j]
        \left(
            \frac{u_z[i+1,j]_{t} - 2u_z[i,j]_{t} + u_z[i-1,j]_{t}}{
                \Delta x^2}
        \right)
    \\[0.3cm]&
    \left.
        +
        (\lambda[i,j] + \mu[i,j])
        \left(
            \frac{u_x[i,j]_{t} - u_x[i-1,j]_{t} - u_x[i,j-1]_{t} +
            u_x[i-1,j-1]_{t}
            }{\Delta x\Delta z}
        \right)
    \right\rbrace
    \end{align*}


----
"""
import numpy

import fatiando.log

log = fatiando.log.dummy('fatiando.seis.wavefd2d')


def elastic_sh():
    """
    """
    pass
