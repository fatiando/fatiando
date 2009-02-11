/* 
* Filename:     macros.h
* Description:  this file contains some general macros used by functions & classes defined in other src files
*
*/

#ifndef _MACROS_H_       
#define _MACROS_H_


/* Macro to help using single pointer variables to wrok with bi-dimensional arrays */
#define POS(i,j,ncols) (((i)*(ncols))+(j)) 


#endif /* _MACROS_H_ */