"""
SimpleTom:
    A simplified Cartesian tomography problem. Does not consider reflection or 
    refraction.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Apr-2010'

import pylab
import numpy
import sys
import time
from PIL import Image
import scipy
from fatiando.directmodels.seismo.simple import traveltime
from fatiando.utils import datamani
from fatiando.math import lu


def random_src_rec(src_n, rec_n, sizex, sizey):
    """
    Make some random sources and receivers. 
    Sources are randomly distributed in the model region (sizex, sizey). 
    Receivers are normally distributed in a circle.
    
    Returns:
        [srcs, recs]
        where:
            srcs = [[xs1,ys1],[xs2,ys2, ...]
            recs = [[[xr1,yr1],[xr2,yr2, ...]        
    """
    
    # Generate the sources and receivers
    srcs_x = numpy.random.random(src_n)*sizex
    srcs_y = numpy.random.random(src_n)*sizey
    srcs = numpy.array([srcs_x, srcs_y]).T
    
    recs_r = numpy.random.normal(0.48*sizex, 0.02*sizex, rec_n)
    recs_theta = numpy.random.random(rec_n)*2*numpy.pi
    recs_x = 0.5*sizex + recs_r*numpy.cos(recs_theta)
    recs_y = 0.5*sizey + recs_r*numpy.sin(recs_theta)
    recs = numpy.array([recs_x, recs_y]).T
    
    return [srcs, recs]


def shoot_rays(src_n, rec_n, model):
    """
    Creates random sources and receivers and shoots rays through the model.
    Sources are randomly distributed over the model and receivers are 
    distributed in a circle around the center of the model. The cell size of the
    model is supposed to be 1.
    
    Returns:
        sources, receivers, traveltimes
        
    sources[i] and receivers[i] are the pair whose ray traveled in traveltime[i]
    """
    
    sizex = len(model[0])
    sizey = len(model)
    
    # Make the sources and receivers
    srcs, recs = random_src_rec(src_n, rec_n, sizex, sizey)
    
    # Compute the travel times
    traveltimes = []
    sources = []
    receivers = []
    for src in srcs:
        for rec in recs:        
            
            time = 0        
            
            for i in range(0, sizey):
                for j in range(0, sizex):
                    
                    time += traveltime(model[i][j], \
                                       j, i, \
                                       j + 1, i + 1, \
                                       src[0], src[1], rec[0], rec[1])
            traveltimes.append(time)
            sources.append(src)
            receivers.append(rec)
            
    return [sources, receivers, traveltimes]



def simple_model(sizex, sizey, slowness_outer=1, slowness_inner=2):
    """
    Make a simple model with a different slowness region in the middle.
    Cuts the model region in cells of 1 x 1.
        
    Parameters:
        
        sizex: size of the model region in the x direction
        
        sizey: size of the model region in the y direction
        
        slowness_outer: slowness of the outer region of the model
        
        slowness_inner: slowness of the inner region of the model
        
    Returns:
    
        model: 2D list with the slowness values of the generated model;
    """
    
    model = slowness_outer*numpy.ones((sizey, sizex))
    
    for i in range(sizey/4, 3*sizey/4 + 1, 1):        
        for j in range(sizex/4, 3*sizex/4 + 1, 1):
            
            model[i][j] = slowness_inner
            
    return model
         
    
def invert(sizex, sizey, data, srcs, recs, reg_param, contam=None):
    
    # Define a discretization for the inversion model
    ############################################################################    
    cellxs = numpy.arange(0., sizex)
    cellys = numpy.arange(0., sizey)    
    ############################################################################
    
    # Build the sensibility matrix
    ############################################################################
    sys.stderr.write("\nBuilding sensibility matrix...")    
    start = time.clock()
    A = []
    for i in range(len(data)):
            
        line = []
    
        for y in cellys:
            for x in cellxs: 
                
                line.append(traveltime(1, \
                                       x, y, \
                                       x + 1, y + 1, \
                                       float(srcs[i][0]), float(srcs[i][1]), \
                                       float(recs[i][0]), float(recs[i][1])))
                
        A.append(line)
    
    end = time.clock()
    sys.stderr.write(" Done (%g s)\n" % (end-start))
    ############################################################################
    
    # Use Tikhonov regularization to solve the inverse problem
    ############################################################################
    sys.stderr.write("\nInverting with Tikhonov 0 regularization...\n")
    sys.stderr.write("  Reg param: %g\n" % (reg_param))
    start = time.clock()
    
    A = numpy.matrix(A)
    
    # Check if the problem is under or over determined
    if len(data) >= A.shape[1]:
        
        # The OVERDETERMINED case
        sys.stderr.write("  Solving an OVERDETERMINED system\n")
        sys.stderr.write("  Building Normal Equations...")
        start_minor = time.clock()
        
        N = A.T*A + reg_param*numpy.matrix(numpy.identity(sizex*sizey))
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))
        
#        sys.stderr.write("  Determinant of normal equations: %g\n" \
#                         % (numpy.linalg.det(N)))
        
        sys.stderr.write("  Performing LU decomposition...")
        start_minor = time.clock()
        
        LU, permut = lu.decomp(N.tolist())
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))
        
        y = A.T*numpy.matrix(data).T
        
        sys.stderr.write("  Solving the linear system...")
        start_minor = time.clock()
        
        est_model = lu.solve(LU, permut, y.T.tolist()[0]) 
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))            
        
        # Contaminate the model with errors 'contam' number of times
        estimates = [est_model]
        if contam != None:
            sys.stderr.write("\nContaminating data %d times\n" % (contam))
            for i in range(contam):
                
                contam_data, stddev = datamani.contaminate(data, 0.01, \
                                                           percent=True, \
                                                           return_stddev=True)
                
                y = A.T*numpy.matrix(contam_data).T
                
                est_model = lu.solve(LU, permut, y.T.tolist()[0]) 
        
                estimates.append(est_model)
                
            sys.stderr.write("  Standard deviation: %g\n" % (stddev))
            
    else:              
        # The UNDERDETERMINED case
        sys.stderr.write("  Solving an UNDERDETERMINED system\n")
        sys.stderr.write("  Building Normal Equations (N)...")
        start_minor = time.clock()
        
        N = A*A.T + reg_param*numpy.matrix(numpy.identity(len(data)))
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))
        
#        sys.stderr.write("  Determinant of N: %g\n" \
#                         % (numpy.linalg.det(N)))
        
        sys.stderr.write("  Performing LU decomposition...")
        start_minor = time.clock()
        
        LU, permut = lu.decomp(N.tolist())
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))
        
        sys.stderr.write("  Calculating N inverse...")
        start_minor = time.clock()
        
        Ninv = lu.inv(LU, permut)
        Ninv = numpy.matrix(Ninv)
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))            
                
        sys.stderr.write("  Calculating the estimate...")
        start_minor = time.clock()
        
        est_model = A.T*Ninv*numpy.matrix(data).T
        
        end_minor = time.clock()
        sys.stderr.write("  Done (%g s)\n" % (end_minor - start_minor))    
        
        # Contaminate the model with errors 'contam' number of times
        estimates = [est_model.T.tolist()[0]]
        if contam != None:
            sys.stderr.write("\nContaminating data %d times\n" % (contam))
            for i in range(contam):
                
                contam_data, stddev = datamani.contaminate(data, 0.01, \
                                                           percent=True, \
                                                           return_stddev=True)
                                                
                est_model = A.T*Ninv*numpy.matrix(contam_data).T
        
                estimates.append(est_model.T.tolist()[0])
                
            sys.stderr.write("  Standard deviation: %g\n" % (stddev))
    ############################################################################
    
    ############################################################################
    # Do some statistics on the estimates
    sys.stderr.write("\nComputing model statistics\n")
    estimates = numpy.array(estimates)
    est_model = []
    est_model_stddev = []
    for param in estimates.T:
        mean = param.mean()
        stddev = param.std()
        
        est_model.append(mean)
        est_model_stddev.append(stddev)
    
    # Compute the residuals
    sys.stderr.write("\nComputing residuals\n")
    residuals = numpy.matrix(data).T - A*numpy.matrix(est_model).T
    residuals = residuals.T.tolist()[0]
    
    # Transform the estimated model into a matrix
    est_model= [est_model[i:i+sizex] for i in range(0, len(est_model), sizex)]
    est_model_stddev = [est_model_stddev[i:i+sizex] \
                        for i in range(0, len(est_model_stddev), sizex)]
    
    end = time.clock()
    sys.stderr.write("  Done (%g s)\n\n" % (end-start))
    ############################################################################


    return est_model, est_model_stddev, residuals
    

def main(src_n=15, rec_n=10, reg_param=0.01, modeltype='simple', \
         sizex=10, sizey=10, cmap=pylab.cm.Greys):
    
    sys.stderr.write("SimpleTom: simple Cartesian tomography\n\n")
    total_start = time.clock()
    
    
    if modeltype == 'simple':
        
        # Make a simple synthetic model
        ########################################################################
        sys.stderr.write("Using a simple synthetic model\n")            
        model = simple_model(sizex, sizey, slowness_outer=1, slowness_inner=2)
        ########################################################################
    
    else:
    
        # Load an image as a model
        ########################################################################
        sys.stderr.write("Loading model from image file '%s'\n" % (modeltype))    
        im = Image.open(modeltype)
        imagearray = scipy.misc.fromimage(im, flatten=True)
        model = (numpy.max(imagearray) + 1 - imagearray)/numpy.max(imagearray)
        model = model.tolist()
        model.reverse()
        model = numpy.array(model)
        sizey, sizex = model.shape
        ########################################################################
            
    sys.stderr.write("  Model size: %d x %d" % model.shape)
    sys.stderr.write(" = %d\n" % (sizex*sizey))     
    
    # Shoot rays
    ############################################################################ 
    sys.stderr.write("\nShooting rays...")    
    start = time.clock()
    srcs, recs, data = shoot_rays(src_n, rec_n, model)
    
    srcs = numpy.array(srcs)
    recs = numpy.array(recs)
       
    end = time.clock()
    sys.stderr.write(" Done (%g s)\n" % (end-start))
    sys.stderr.write("  Sources: %d  Receivers: %d\n" % (src_n, rec_n))
    sys.stderr.write("  Number of data: %d\n" % (len(data)))
    ############################################################################
    
    # Invert
    ############################################################################
    
    estimate, stddevs, residuals = invert(sizex, sizey, data, srcs, recs, \
                                          reg_param=reg_param, contam=10)
    
    ############################################################################
    
    
    # Plot the model with the sources and receivers
    ############################################################################
    sys.stderr.write("Plotting sources and receivers...\n")        
    pylab.figure()
    pylab.axis('scaled')
    pylab.title('True Model with Sources and Receivers')
    
    pylab.pcolor(model, cmap=cmap, \
                 vmin=numpy.min(model), vmax=numpy.max(model))
    cb = pylab.colorbar(orientation='vertical')
    cb.set_label("Slowness")
    
    pylab.plot(srcs.T[0], srcs.T[1], 'r*', ms=9, label='Source')
    pylab.plot(recs.T[0], recs.T[1], 'b^', ms=8, label='Receiver')
    
    pylab.legend(numpoints=1, prop={'size':7})
             
    pylab.xlim(0, sizex)
    pylab.ylim(0, sizey)   
    ############################################################################
    
    # Plot the raypaths
    ############################################################################
    sys.stderr.write("Plotting raypaths...\n")        
    pylab.figure()
    pylab.axis('scaled')    
    pylab.title('Raypaths')    
    
    for i in range(len(data)):
        pylab.plot([srcs[i][0],recs[i][0]], [srcs[i][1],recs[i][1]], 'k-')      
    
    pylab.plot(srcs.T[0], srcs.T[1], 'r*', ms=9, label='Source')
    pylab.plot(recs.T[0], recs.T[1], 'b^', ms=8, label='Receiver')
    
    pylab.legend(numpoints=1, prop={'size':7})
             
    pylab.xlim(0, sizex)
    pylab.ylim(0, sizey)
    ############################################################################
    
    # Plot a histogram of the travel times    
    ############################################################################
    sys.stderr.write("Plotting traveltimes...\n")
    pylab.figure()
    pylab.hist(data, bins=len(data)/8, facecolor='gray')
    pylab.title("Travel times")
    pylab.xlabel("Travel time")
    pylab.ylabel("Count")
    ############################################################################
    
    # Plot the result and a histogram of the residuals
    ############################################################################
    sys.stderr.write("Plotting results...\n")
    pylab.figure()
    pylab.hist(residuals, bins=len(residuals)/8, facecolor='gray')
    pylab.title("Residuals")
    pylab.xlabel("Residuals")
    pylab.ylabel("Count")
    
    pylab.figure()
    pylab.axis('scaled')
    pylab.title('Inversion Result')    
    pylab.pcolor(numpy.array(estimate), cmap=cmap, \
                 vmin=numpy.min(model), vmax=numpy.max(model))
    cb = pylab.colorbar(orientation='vertical')
    cb.set_label("Slowness")                   
    pylab.xlim(0, sizex)
    pylab.ylim(0, sizey)
    
    pylab.figure()
    pylab.axis('scaled')
    pylab.title('Result Standard Deviation')    
    pylab.pcolor(numpy.array(stddevs), cmap=pylab.cm.jet)
    cb = pylab.colorbar(orientation='vertical')
    cb.set_label("Standard Deviation")          
    pylab.xlim(0, sizex)
    pylab.ylim(0, sizey)
    ############################################################################
    
    total_end = time.clock()    
    sys.stderr.write("\nFinished all in %g s\n" % (total_end - total_start))
    
    pylab.show()
    
    
if __name__ == '__main__':
    
    main(src_n=70, rec_n=40, reg_param=1, \
         modeltype='/home/leo/src/fatiando/examples/simpletom/mickey3.jpg', \
         cmap=pylab.cm.Greys)


#    main(src_n=60, rec_n=20, sizex=50, sizey=50, reg_param=0.01, modeltype='simple')
    
