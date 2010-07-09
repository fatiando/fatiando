/* **************************************************************************

   This module contains a set of functions that calculate the travel times in a
   simplified tomography with no reflection or refraction.

   Author: Leonardo Uieda
   Date: 29 April 2010
   Last Update: $DATE: $
   $REVISION: $

   ************************************************************************** */


#ifndef _SIMPLETOM_H_
#define _SIMPLETOM_H_


/* MACROS */
/* ************************************************************************** */

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* ************************************************************************** */

/* FUNCTION DECLARATIONS */
/* ************************************************************************** */

/* Returns the travel time inside a given cell.
 Parameters:
    * double slowness: the slowness of the cell. Must be in units compatible
        with the other parameters!
    * double x1, y1: coordinates of the lower-left corner of the cell
    * double x2, y2: coordinates of the upper-right corner of the cell
    * double x_src, y_src: coordinates of the wave source
    * double x_rec, y_rec: coordinates of the receiver
*/
extern double traveltime(double slowness, double x1, double y1,
                         double x2, double y2,
                         double x_src, double y_src,
                         double x_rec, double y_rec);

/* ************************************************************************** */

#endif
