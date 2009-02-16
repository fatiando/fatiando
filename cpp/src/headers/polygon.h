/* 
* Filename:     polygon.h
* Description:  this file basically contains the Polygon class
*
*/

#ifndef _POLYGON_H_       
#define _POLYGON_H_


#include <malloc.h>
#include <stdio.h>
#include "macros.h"


/*
* Class:      Polygon
* Summary:    the main purpose of this class is to work as an interface between the GUI class(es) and the modelling algorithms, 
*             providing the resources for creating 2D physical models (polygons), and providing an easy-to-use set of graphical
*             parameters.
* Properties: 
* Methods:    
* Examples:   
*/

class Polygon
{
    private:
        
        double     *vertices; /* 2-dimensional array containing the vertices */
        long int  n_vertices; /* number of vertices */
        
        /* the graphical params might need some changes, depending on the UI lib */
        float      color_red;
        float    color_green;
        float     color_blue;
        int       line_style;
        
        /* physical properties */
        double       density;
        double magnetization;
        /* etc */

        /* This function is called to reallocate *vertices' memory. This could be replaced by a macro, eventually */
        void redim(long int new_size)
        {
            n_vertices = new_size;
            vertices = (double *) realloc(vertices, sizeof(double) * (n_vertices+1) * 2 );
        }
        
        /* Given the vertix coordinates, this function returns it's index (or -1, if the vertix doesn't exist) */
        long int get_index(double x_coord, double y_coord)
        {
            long int i;
            
            for (i = 0 ; i < n_vertices ; i++ )
                if ( ( vertices[POS(i,0,2)] == x_coord ) && ( vertices[POS(i,1,2)] == y_coord ) ) return i;
                
            return -1;
        }
        
        
    public:

        /* prints the *vertices array for debbuging purposes (will be removed eventually) */
        void print(void)
        {
            long int i;
            
            fprintf(stdout, "index\t X   \t Y\n");
            for (i = 0 ; i < n_vertices ; i++)
            {
                fprintf(stdout, "%ld:   \t%2.2lf\t%2.2lf\n", i, vertices[POS(i,0,2)], vertices[POS(i,1,2)]);
            }
            fprintf(stdout,"\n");
        }
        
        /* Main constructor */
        Polygon(void)
        {
                n_vertices = 0;
                vertices = (double *) malloc(sizeof(double) * (2)); /* allocate the space for 1 vertix */
        }
        
        /* ToDo: create the oprerators' overloads */
        
        
        /* basic methods for the class: */
        
        /* This function adds a vertix at the end of the polygon (after the last vertix). */
        void add_vertix( double x_coord, double y_coord )
        {
            /* ToDo: check if the polygon is already closed */
            
            /* ToDo: check if the given coordinates will create a line crossing a previous side of the polygon */
            
            /* add another line to *vertices */
            redim(n_vertices+1);
            
            vertices[POS(n_vertices-1, 0, 2)] = x_coord;
            vertices[POS(n_vertices-1, 1, 2)] = y_coord;
        }
        
        
        /* This function adds a vertix into a specific index of the polygon (pushing down all vertices one row). */
        void add_vertix( double x_coord, double y_coord, long int index )
        {
            long int i;
            
            /* ToDo: check if the given coordinates will create a line crossing a previous side of the polygon */
            
            redim(n_vertices+1);
            
            /* push down all vertices */
            for (i = n_vertices - 1 ; i > index; i--)
            {
                vertices[POS(i, 0, 2)] = vertices[POS(i-1, 0, 2)];
                vertices[POS(i, 1, 2)] = vertices[POS(i-1, 1, 2)];
            }
            
            /* add the new one */
            vertices[POS(index, 0, 2)] = x_coord;
            vertices[POS(index, 1, 2)] = y_coord;
        }
        
        
        /* Removes a specific vertix from the vertices array (using vertix index) */
        void rem_vertix( long int index )
        {
            long int i;
            
            /* ToDo: check if index < n_vertices */
            /* ToDo: if the polygon is closed, check if there're more than 3 vertices, and if they're on the same line */
            
            /* push all vertices upwards */
            for (i = index ; i < n_vertices ; i++ )
            {
                vertices[POS(i, 0, 2)] = vertices[POS(i+1, 0, 2)];
                vertices[POS(i, 1, 2)] = vertices[POS(i+1, 1, 2)];
            }
            
            /* remove unnecessary allocated mem */
            redim(n_vertices-1);
        }
        
        
        /* Removes a specific vertix from the vertices array (using the coordinates) */
        void rem_vertix( double x_coord, double y_coord )
        {
            long int index;
            
            index = get_index(x_coord, y_coord);
            rem_vertix(index);
        }

        
        /* Edit (replace) the vertix on "index" with a new one */
        void edt_vertix ( long int index, double x_coord, double y_coord )
        {
            /* ToDo: check if the given coordinates will create a line crossing a previous side of the polygon */
            /* ToDo: if the polygon was closed, check if index == n_vertices || index == 0 */
            vertices[POS(index, 0, 2)] = x_coord;
            vertices[POS(index, 1, 2)] = y_coord;
        }
        
        
        /* now some methods to check the polygon consistency (is it closed? is it complex? ...)*/
        
        /* Only checks if first vertix == last vertix. */
        int is_closed(void)
        {
            return ((vertices[POS(0, 0, 2)] == vertices[POS(n_vertices-1, 0, 2)]) && (vertices[POS(0, 1, 2)] == vertices[POS(n_vertices-1, 1, 2)]));
        }
        
        
        
};


#endif /* _POLYGON_H_ */




