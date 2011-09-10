/* *****************************************************************************
 Copyright 2010 The Fatiando a Terra Development Team

 This file is part of Fatiando a Terra.

 Fatiando a Terra is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Fatiando a Terra is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************** */

/* **************************************************************************

 Functions that calculate the travel times of seismic waves.

 Author: Leonardo Uieda
 Date: 29 April 2010

 **************************************************************************** */


#ifndef _TRAVELTIME_H_
#define _TRAVELTIME_H_


/* MACROS */
/* ************************************************************************** */

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* ************************************************************************** */

/* FUNCTION DECLARATIONS */
/* ************************************************************************** */

/* Calculate the travel time inside a 2D square cell assuming the ray is a
   straight line.

 Parameters:
    * double slowness: the slowness of the cell. Must be in units compatible
        with the other parameters!
    * double x1, y1: coordinates of the lower-left corner of the cell
    * double x2, y2: coordinates of the upper-right corner of the cell
    * double x_src, y_src: coordinates of the wave source
    * double x_rec, y_rec: coordinates of the receiver
*/
extern double straight2d(double slowness, double x1, double y1, double x2,
	double y2, double x_src, double y_src, double x_rec, double y_rec);

/* ************************************************************************** */

#endif
