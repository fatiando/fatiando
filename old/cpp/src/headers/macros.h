/* 
 * Filename:     macros.h
 * $Revision$
 * Last edited: $Date$
 * Edited by: $Author$
 *
 * Created by: Zezinho
 *
 * Description:
 *  This file contains some general macros used by functions & classes defined
 *  in other src files.
 */

#ifndef _MACROS_H_       
#define _MACROS_H_


/* Macro to help using single pointer variables to wrok with bi-dimensional arrays */
#define POS(i,j,ncols) (((i)*(ncols))+(j))
/* Macros to calculate the maximum and minimum of two numbers */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


#endif /* _MACROS_H_ */