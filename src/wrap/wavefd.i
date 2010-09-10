/* *****************************************************************************
 Copyright 2010 Leonardo Uieda

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
   Interface file for generation SWIG wrappers around 'wavefd.c'

   OBS: Be careful with the typemaps! Using different types with the same
        argument names will cause wierd problems! 
   ************************************************************************** */
   
      
/* The module docstring */
%define DOCSTRING
"
_wavefd_ext:

    C coded time steps for finite differences solvers for the elastic wave 
    equation.    

Author: Leonardo Uieda
Created 29 June 2010
"
%enddef
/* Declare the module name */
%module(docstring=DOCSTRING) wavefd_ext


/* Put the headers with the definitions */
%{

#include "../c/wavefd.h"
#include "typeconversions.c"

%}


/* TIMESTEP1D */
/* ************************************************************************** */
%typemap(in) (double *)
{    
    int tmpsize;

    $1 = PyList_to_vector($input, &tmpsize);
    
    /* Check if there was any error in the conversion */
    if(!$1)
    {
        return NULL;
    }
}

%typemap(in) (double *vel, int size, double *u_tp1)
{    
    $1 = PyList_to_vector($input, &$2);

    /* Check if there was any error in the conversion */
    if(!$1)
    {     
        return NULL;
    }
    
    /* Malloc for the output buffer */
    $3 = (double *)malloc($2*sizeof(double));
        
    /* Check if there was any error in the conversion */
    if(!$3)
    {
       return NULL;
    }    
}

%typemap(argout) (double deltax, double deltat, double *u_tm1, double *u_t, 
        double *vel, int size, double *u_tp1)
{    
    if(!result)
    {
        free($3);
        free($4);
        free($5);
        free($7);     
        PyErr_SetString(PyExc_MemoryError, "Memory allocation problem.");
        return NULL;
    }

    /* Blow away any previous result */
    Py_XDECREF($result);

    $result = vector_to_PyList($7, result);
    
    free($3);
    free($4);
    free($5);
    free($7);
}

%feature("autodoc", "1");
%define TIMESTEP1D_DOC
"
Time step for the 1D finite differences solver of wave equation.

Parameters:

    u_t: amplitude array at time t
    u_tm1: amplitude array at time t - 1
    vel: velocity array (contains the velocities in each node)
    deltax: grid spacing
    deltat: time interval
    
Returns u at time t + 1
OBS: does not set boundary conditions or generator functions!
"
%enddef

%feature("docstring", TIMESTEP1D_DOC);

%rename(timestep1d) timestep1d;

extern int timestep1d(double deltax, double deltat, double *u_tm1, double *u_t,
        double *vel, int size, double *u_tp1);
/* ************************************************************************** */
