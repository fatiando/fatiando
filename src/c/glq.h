/* **************************************************************************

   Set of functions for numerical the integration via Gauss-Legendre Quadrature
   (GLQ).

   Author: Leonardo Uieda
   Date: 07 May 2010

   ************************************************************************** */

#ifndef _GLQ_H_
#define _GLQ_H_


/* CONSTANTS */
/* ************************************************************************** */

#define GLQ_MAXIT 1000

#define GLQ_MAXERROR 0.000000000000001

#ifndef PI
#define PI 3.141592653589793116
#endif

/* ************************************************************************** */


/* MACROS */
/* ************************************************************************** */

/* Macro to calculate the absolute value of x */
#ifndef ABSOLUTE
#define ABSOLUTE(x) ((x) > 0 ? (x) : (-1)*(x))
#endif

/* ************************************************************************** */


/* FUNCTION DECLARATIONS */
/* ************************************************************************** */

/* GLQ_NODES */
/* ************************************************************************** */
/* Calculates the GLQ nodes. The nodes are the roots of the Legendre polynomial
   of degree 'order'. The roots are calculated using Newton's Method for
   Multiple Roots (Barrera-Figueroa et al., 2006).

   REMEMBER: pre-allocate the memory for the buffer!

   Parameters:
       order: order of the GLQ
       nodes: output buffer for the nodes

   Returns:
       'order' if all went well
       0 if there was stagnation when looking for the roots
*/
extern int glq_nodes(int order, double *nodes);


/* GLQ_WEIGHTS */
/* ************************************************************************** */
/* Calculates the weights for the GLQ.

   REMEMBER: pre-allocate the memory for the buffer!

   Parameters:
       order: order of the GLQ
       nodes: pointer to array with the UN-SCALED nodes
       weights: output buffer for the weights

   Returns:
       'order' if all went well
       0 if fails
*/
extern int glq_weights(int order, double *nodes, double *weights);


/* GLQ_SCALE_NODES */
/* ************************************************************************** */
/* Scales the GLQ nodes to the integration interval [a,b]. This is necessary
   because the output of glq_nodes is scaled to [-1,1] (which is the value
   gle_weights needs).

   REMEMBER: pre-allocate the memory for the buffer!

   Parameters:
       a: lower integration limit
       b: upper integration limit
       order: order of the GLQ
       nodes: pointer to array with the UN-SCALED nodes
       scaled: output buffer for the scaled nodes

   Returns:
       'order' if all went well
       0 if fails
*/
extern int glq_scale_nodes(double a, double b, int order, double *nodes,
                           double *scaled);

/* ************************************************************************** */

#endif
