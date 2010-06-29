/* **************************************************************************

   Time steps for a finite difference solver to the elastic wave equation.

   Author: Leonardo Uieda
   Date: 29 June 2010

   ************************************************************************** */

#ifndef _WAVEFD_H_
#define _WAVEFD_H_


/* Perform a time step for 1D finite differences.
 *
 * Parameters:
 *
 *      u_t: amplitude array at time t
 *      u_tm1: amplitude array at time t - 1
 *      vel: velocity array (contains the velocities in each node)
 *      size: number of nodes in the grid
 *      deltax: grid spacing
 *      deltat: time interval
 *      u_tp1: amplitude array at time t + 1 (return values)
 * */
extern int timestep1d(double deltax, double deltat, double *u_tm1, double *u_t,
        double *vel, int size, double *u_tp1);


#endif
