/* **************************************************************************

   Set of functions for performing the LU decomposition of a matrix.

   OBS: All matrices are treated as 1-dimensional double arrays and are accessed
        using the macro POS in lu.h

   Author: Leonardo Uieda
   Date: 01 March 2010

   ************************************************************************** */

#ifndef _LU_H_
#define _LU_H_

/* MACROS */
/* ************************************************************************** */

/* Macro for accessing the element i,j of a matrix with ncols columns. This
matrix should be laid out in a 1D array. */
#define POS(i,j,ncols) (((i)*(ncols)) + (j))

/* Macro to calculate the absolute value of x */
#define ABSOLUTE(x) ((x) > 0 ? (x) : (-1)*(x))

/* ************************************************************************** */


/* FUNCTION DECLARATIONS */
/* ************************************************************************** */

/* LU_DECOMP_NOPIVOT */
/* Does the LU decomposition of a square matrix of dimensions 'dim' WITHOUT
   pivoting. Returns the result in the previously allocated matrix 'lu'. Matrix
   U is given in the upper triangle and L in the lower. The diagonal of L (=1)
   is omitted. REMEMBER: pre-allocate the memory for the buffers!
   Parameters:
       matrix: pointer to a 1D array containing a SQUARE matrix such as
               [a11,a12,a13,...,a1N,a21,a22,...,a2N,...,aNN]
       dim: the dimension N of the matrix
       lu: the'dim'x'dim' sized buffer where the L and U matrices will be
           returned.
   Returns:
       'dim' if all went well;
       0 if the matrix is not decomposable

   NOTE: The pivoting alternative to this function is prefered because it was
   more extensively tested. See lu_decomp_pivot documentation for more details.
*/
extern int lu_decomp_nopivot(double *matrix, int dim, double *lu);


/* LU_DECOMP */
/* Does the LU decomposition of a square matrix of dimensions 'dim' using
   pivoting. Returns the result in the previously allocated vector 'lu'. Matrix
   U is given in the upper triangle and L in the lower. The diagonal of L (=1)
   is omitted. The permutations are returned in 'permut' (line i was permuted
   with line permut[i]). REMEMBER: pre-allocate the memory for the buffers!
   Parameters:
       matrix: pointer to a 1D array containing a SQUARE matrix such as
               [a11,a12,a13,...,a1N,a21,a22,...,a2N,...,aNN]
       dim: the dimension N of the matrix
       lu: the buffer where the L and U matrices will be returned. The diagonal
           elements of L are omitted (=1) and the 2 matrices are joined into one
       permut: the 'dim' sized buffer where the permutations will be returned.
   Returns:
       'dim' if all went well;
       0 if the matrix is not decomposable
*/
extern int lu_decomp(double *matrix, int dim, double *lu, int *permut);


/* SOLVE_LU */
/* Solve a linear system Ax=y given it's LU decomposition (with pivoting).
 * REMEMBER: pre-allocate the memory for the buffers!
 * Parameters:
 *      lu: pointer to a 1D array containing the LU decomposition of a matrix
 *          such as returned by lu_decomp
 *          EX: [u11,u12,u13,...,u1N,l21,u22,...,u2N,...,uNN]
 *      dim: the dimension N of the matrix
 * 	    permut: pointer to a 1D array with the permutations done in the LU
 * 	       decomposition as returned by lu_decomp
 *      y: pointer to a 1D array with the ordinate values;
 *      x: pointer to a 1D array with the output buffer
 *  Returns:
 *      'dim' if all went well
 *      0 if an error occurred
 */
extern int solve_lu(double *lu, int dim, int *permut, double *y,
                           double *x);
/* ************************************************************************** */

#endif
