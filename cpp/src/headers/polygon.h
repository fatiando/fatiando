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
*
*   add_vertix( <x> , <y> [, <index>] )
*   rem_vertix( [ <x> , <y> ] | [ <index> ] )
*   edt_vertix ( <index> , <x> , <y> )
*   is_closed()
*   
* Examples:   
*/

/* This macros probably would suit better inside macros.h */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


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
        double vertix_radius; /* i'm still thinking about this one: Maybe we should add circles at the vertices to make them more accessible */
        
        
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
        

        /* Checks if the new vertix (x_coord, y_coord) is valid (i.e. if it does not create a crossing   */
        /* line with previous lines/vertices).  This particular function only works for the last vertix, */
        /* which means this function should be called while adding a new vertix at the "end"             */
        /* of the polygon. Returns 1 if it is valid, 0 otherwise.                                        */
        int is_valid_vertix( double x_coord, double y_coord  )
        {
            long int i; /* used for indexing */
            /* and here are a lot of variables used to handle the lines' coefficients. */
            /* these could be removed to improve the speed (there're a lot of unecessary attributions) */
            double x0, x1, y0, y1, a0, b0;
            double xn, yn, a, b; /* last vertix, and the coefficients of the line created between (xn,yn) and (x_coord,y_coord) */
            double x_inter, y_inter; /* intersection coordinates */
            
            xn = vertices[POS(n_vertices-1, 0, 2)];
            yn = vertices[POS(n_vertices-1, 1, 2)];
            
            /* starts checking if the new line is vertical (if it is, the whole problem changes).          */
            /* this comparision could also be made by considering numerical precision, since close numbers */
            /* might result an almost-vertical line. This has to be tested eventually                      */
            if ( xn == x_coord )
            {
                for (i = 0; i < n_vertices; i++)
                {
                    x0 = vertices[POS(i, 0, 2)]; x1 = vertices[POS(i+1, 0, 2)];
                    y0 = vertices[POS(i, 1, 2)]; y1 = vertices[POS(i+1, 1, 2)];
                
                    /* same "numerical precision" bullshit here */
                    if ( x0 == x1 )
                    {
                        /* here we have 2 vertical lines, so, the only way they could intersect is by */
                        /* having the same x coordinate, and being at the same interval               */
                        if (x0 == xn)
                        {
                            /* now we have some possibilities... let's start with the great exception: coincident lines */
                            /* this case would only occur if the polygon is already closed, but i'm leaving this here   */
                            /* for now.                                                                                 */
                            if ((MIN(y0,y1) == MIN(y_coord,yn)) && (MAX(y0,y1) == MAX(y_coord,yn)))
                            {
                                return 0;
                            }
                            else if () /* gotta think about this */
                            {
                                /* hm */
                            }
                        }
                    }
                }
            }
            else /* And here's the default routine (for a non-vertical new line) */
            {
                a = (y_coord - yn)/(x_coord - xn);
                b = yn - (a * xn);
                
                for (i = 0; i < n_vertices; i++)
                {
                    x0 = vertices[POS(i, 0, 2)]; x1 = vertices[POS(i+1, 0, 2)];
                    y0 = vertices[POS(i, 1, 2)]; y1 = vertices[POS(i+1, 1, 2)];
                    
                    /* same "numerical precision" bullshit here */
                    if ( x0 == x1 )
                    {
                        /* here we have a vertical line, so, we basically have to check the intervals */
                        if (( MIN(x_coord, xn) <= x0 ) && ( x0 <= MAX(x_coord, xn) ))
                        {
                            y_inter = (a * x0) + b;
                            if(( MIN(y_coord, yn) <= y_inter ) && ( y_inter <= MAX(y_coord, yn) ))
                            {
                                return 0; /* the lines are crossing... */
                            }
                        }
                    }
                    else /* default */
                    {
                        a0 = (y1 - y0)/(x1 - x0);
                        b0 = y0 - ( a0 * x0);
                        
                        /* if they're parallel there's no reason to check if they cross */
                        if (a0 != a) 
                        {
                            y_inter = ((-b0) + ((a0 / a) * b)) / ((a0/a) - 1);
                        
                            /* Now we check if the intersection point is within both intervals */
                            if (((MIN(y0,y1) <= y_inter) && (y_inter <= MAX(y0,y1)))  &&  ((MIN(yn,y_coord) <= y_inter) && (y_inter <= MAX(yn,y_coord))))
                            {
                                return 0; /* ok, the lines are crossing */
                            } 
                        }
                    }
                } 
            }
            
            return 1; /* if it has come this far, then it's ok: there're no lines crossing with the new one */
        }
        
};


#endif /* _POLYGON_H_ */




