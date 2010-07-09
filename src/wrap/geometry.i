%module geometry

%{

#include "../c/geometry.c"

%}

%include "../c/geometry.c"

%extend Point3D {

    Point3D(double x=0, double y=0, double z=0)
    {
        Point3D *p;

        p = (Point3D *)malloc(sizeof(Point3D));

        p->x = x;
        p->y = y;
        p->z = z;

        return p;
    }

    ~Point3D()
    {
        free(self);
    }
}

%extend Prism {

    Prism(double dens=0, double x1=0, double x2=0, double y1=0, double y2=0, 
          double z1=0, double z2=0)
    {
        Prism *prism;

        prism = (Prism *)malloc(sizeof(Prism));

        prism->dens = dens;
        prism->x1 = x1;
        prism->x2 = x2;
        prism->y1 = y1;
        prism->y2 = y2;
        prism->z1 = z1;
        prism->z2 = z2;

        return prism;
    }

    ~Prism()
    {
        free(self);
    }
}

%extend Tesseroid {

    Tesseroid(double dens=0, double north=0, double south=0, double east=0, 
              double west=0, double z1=0, double z2=0)
    {
        Tesseroid *tess;

        tess = (Tesseroid *)malloc(sizeof(Tesseroid));

        tess->dens = dens;
        tess->north = north;
        tess->south = south;
        tess->east = east;
        tess->west = west;
        tess->z1 = z1;
        tess->z2 = z2;

        return tess;
    }

    ~Tesseroid()
    {
        free(self);
    }
}
