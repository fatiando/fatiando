# TODO: Make it nice for Fatiando. Do the linear derivative. Use logging
"""
ClimateChange:
    Invert the heat profile of a well to get the past climate changes.
"""

import numpy
import pylab
from fatiando.directmodels.heat import climatesignal
from fatiando.math import lu
from fatiando.utils.datamani import contaminate
import sys
from time import clock

def invert_linear(temps, depths, mag0, t0, reg_param, marq_param=1, contam=False):
    
    # A few params
    maxit = 1000            # Maximum iterations
    marq_itmax = 10
    I = numpy.identity(2)
    
    sys.stdout.write("\nLinear change in temperature\n\n")
    
    if contam == True:
    
        temps, stddev = contaminate(temps, 0.01, percent=True, return_stddev=True)
        sys.stdout.write("\nContaminado com %g C de erro\n" % (stddev))
        
    # Plota os dados
    pylab.figure()
    pylab.plot(temps, -1*depths, '.k')
    pylab.title("Perfil de temperatura")
    pylab.xlabel("Temperatura (C)")
    pylab.ylabel("Profundidade (m)")
    ############################################################################
      
    # The initial solution
    ############################################################################
    sys.stderr.write("\nInitial solution:\n")
    sys.stderr.write("  mag0: %g C\n" % (mag0))
    sys.stderr.write("  t0: %g years\n" % (t0))
    sys.stdout.write("\nInitial solution:\n")
    sys.stdout.write("  mag0: %g C\n" % (mag0))
    sys.stdout.write("  t0: %g years\n" % (t0))
    p1 = numpy.array([mag0, t0])
    ############################################################################
    
    # Compute the goal function for initial solution
    ############################################################################
    goal = []
    residuals = []
    adjusted_temps = []
    tmpgoal = 0.0
    for i in range(len(temps)):
        
        adjusted_temps.append(climatesignal.linear(depths[i], p1[0], p1[1]))
        
        residuals.append(temps[i] - adjusted_temps[i])
        
        tmpgoal += residuals[i]**2
        
    goal.append(tmpgoal)    
    residuals = numpy.matrix(residuals).T
    ############################################################################
    
    # Plot the initial solution
    ############################################################################
    pylab.figure()
    pylab.plot(temps, -1*depths, '.k', label='Medidos')
    pylab.plot(adjusted_temps, -1*depths, '.-r', label='Ajustados')
    pylab.legend(numpoints=1, prop={'size':7})
    pylab.title('Solucao Inicial')
    pylab.xlabel('Temperatura (C)')
    pylab.ylabel('Profundidade (m)')
    ############################################################################
    
        
    # Iterate to find the minimum goal
    ############################################################################
    sys.stderr.write("\nStarting iteration to find the solution:\n")
    totalstart = clock()
    dt = 0.1
    for iteration in range(1, maxit+1):
        
        sys.stderr.write("\niteration %d:\n" % (iteration))
        
        p0 = p1
        
        # Build the sensibility matrix
        sys.stderr.write("  Compute Jacobian matrix...")
        J = []
        for z in depths:
            
            a0 = climatesignal.linear(z, 1., p0[1])
            
            a1 = (climatesignal.linear(z, p0[0], p0[1]+0.5*dt) - \
                  climatesignal.linear(z, p0[0], p0[1]-0.5*dt))/dt
        
            J.append([a0, a1])
            
        J = numpy.matrix(J)
               
    
        # Make the Hessian
        sys.stderr.write("  Compute Hessian matrix...")
        
        H = J.T*J + reg_param*I
            
        # Make the gradient of the goal function
        sys.stderr.write("  Compute gradient of goal...")
        
        grad = J.T*(residuals) - reg_param*numpy.matrix(p0).T
        grad = grad.T.tolist()[0]
        
        # The Marquardt loop
        sys.stderr.write("  Marquardt loop...")
        start = clock()
        for marq_it in range(1, marq_itmax+1):
            
            N = H + marq_param*(numpy.diag(numpy.diag(H)))
            
            LU, permut = lu.decomp(N.tolist())
            
            deltap = lu.solve(LU, permut, grad)
            
            p1 = p0 + numpy.array(deltap)
            
            # Calculate the goal function
            tmpgoal = 0.0
            for i in range(len(temps)):
                
                residuals[i,0] = temps[i] - climatesignal.linear(depths[i], \
                                                                 p1[0], p1[1])
                
                tmpgoal += residuals[i,0]**2
                        
                                                
            if tmpgoal >= goal[iteration-1]:
                
                marq_param *= 10
                
            else:
                
                marq_param *= 0.1
                
                break        
                
        goal.append(tmpgoal)                
        end = clock()
        sys.stderr.write(" Done (%g s)\n" % (end-start))            
        sys.stderr.write("  Marquardt iterations = %d\n" % \
                         (marq_it))
        sys.stderr.write("  Goal function = %g\n" % (goal[iteration]))
        sys.stderr.write("  Solution: mag=%g   time=%g\n" % (p1[0], p1[1]))
        
        # Get out of the iterations if there is stagnation
        if abs(goal[iteration] - goal[iteration-1]) < 10**(-10):
                        
            break
        
    # Print the results
    totalend = clock()
    sys.stderr.write("\nFinished (%g s)!\n" % (totalend-totalstart))
    sys.stderr.write("\nResults:\n")
    sys.stderr.write("  iterations: %d\n" % (iteration))
    sys.stderr.write("  Goal function: %g\n" % (goal[iteration]))
    sys.stderr.write("  Magnitude: %g\n" % (p1[0]))
    sys.stderr.write("  Time: %g\n" % (p1[1]))
    sys.stdout.write("\nResults:\n")
    sys.stdout.write("  iterations: %d\n" % (iteration))
    sys.stdout.write("  Goal function: %g\n" % (goal[iteration]))
    sys.stdout.write("  Magnitude: %g\n" % (p1[0]))
    sys.stdout.write("  Time: %g\n" % (p1[1]))
    
    pylab.figure()
    pylab.hist(residuals.T.tolist()[0], \
               bins=len(residuals)/8, \
               normed=False, facecolor='gray', alpha=0.75)
    pylab.title('Residuos')
    pylab.ylabel('Contagem')
    pylab.xlabel('Residuos (C)')
    
    adjusted_temps = []
    for i in range(len(depths)):        
        adjusted_temps.append(climatesignal.linear(depths[i], p1[0], p1[1]))
    
    pylab.figure()
    pylab.plot(temps, -1*depths, '.k', label='Medido')
    pylab.plot(adjusted_temps, -1*depths, '.-r', label='Ajustado')
    pylab.legend(numpoints=1, prop={'size':7})
    pylab.title('Solucao Final')
    pylab.xlabel('Temperatura (C)')
    pylab.ylabel('Profundidade (m)')
    
           
    pylab.figure()
    pylab.plot(goal, '.-k')
    pylab.title('Funcao Objetivo')
    pylab.xlabel('Iteracao')
    pylab.ylabel('Erro')
    pylab.show()
    

def invert_abrupt(temps, depths, mag0, t0, reg_param, marq_param=1, contam=False):
    
    # A few params
    maxit = 1000            # Maximum iterations
    marq_itmax = 10
    I = numpy.identity(2)
    
    sys.stdout.write("\nAbrupt change in temperature\n\n")
    
    if contam == True:
    
        temps, stddev = contaminate(temps, 0.01, percent=True, return_stddev=True)
        sys.stdout.write("\nContaminado com %g C de erro\n" % (stddev))
        
    # Plota os dados
    pylab.figure()
    pylab.plot(temps, -1*depths, '.k')
    pylab.title("Perfil de temperatura")
    pylab.xlabel("Temperatura (C)")
    pylab.ylabel("Profundidade (m)")
    ############################################################################
      
    # The initial solution
    ############################################################################
    sys.stderr.write("\nInitial solution:\n")
    sys.stderr.write("  mag0: %g C\n" % (mag0))
    sys.stderr.write("  t0: %g years\n" % (t0))
    sys.stdout.write("\nInitial solution:\n")
    sys.stdout.write("  mag0: %g C\n" % (mag0))
    sys.stdout.write("  t0: %g years\n" % (t0))
    p1 = numpy.array([mag0, t0])
    ############################################################################
    
    # Compute the goal function for initial solution
    ############################################################################
    goal = []
    residuals = []
    adjusted_temps = []
    tmpgoal = 0.0
    for i in range(len(temps)):
        
        adjusted_temps.append(climatesignal.abrupt(depths[i], p1[0], p1[1]))
        
        residuals.append(temps[i] - adjusted_temps[i])
        
        tmpgoal += residuals[i]**2
        
    goal.append(tmpgoal)    
    residuals = numpy.matrix(residuals).T
    ############################################################################
    
    # Plot the initial solution
    ############################################################################
    pylab.figure()
    pylab.plot(temps, -1*depths, '.k', label='Medidos')
    pylab.plot(adjusted_temps, -1*depths, '.-r', label='Ajustados')
    pylab.legend(numpoints=1, prop={'size':7})
    pylab.title('Solucao Inicial')
    pylab.xlabel('Temperatura (C)')
    pylab.ylabel('Profundidade (m)')
    ############################################################################
    
        
    # Iterate to find the minimum goal
    ############################################################################
    sys.stderr.write("\nStarting iteration to find the solution:\n")
    totalstart = clock()
    for iteration in range(1, maxit+1):
        
        sys.stderr.write("\niteration %d:\n" % (iteration))
        
        p0 = p1
        
        # Build the sensibility matrix
        sys.stderr.write("  Compute Jacobian matrix...")
        J = []
        for z in depths:
            
            a0 = climatesignal.abrupt(z, 1., p0[1])
            
            a1 = climatesignal.abrupt_time_derivative(z, p0[0], p0[1])
        
            J.append([a0, a1])
            
        J = numpy.matrix(J)
               
    
        # Make the Hessian
        sys.stderr.write("  Compute Hessian matrix...")
        
        H = J.T*J + reg_param*I
            
        # Make the gradient of the goal function
        sys.stderr.write("  Compute gradient of goal...")
        
        grad = J.T*(residuals) - reg_param*numpy.matrix(p0).T
        grad = grad.T.tolist()[0]
        
        # The Marquardt loop
        sys.stderr.write("  Marquardt loop...")
        start = clock()
        for marq_it in range(1, marq_itmax+1):
            
            N = H + marq_param*(numpy.diag(numpy.diag(H)))
            
            LU, permut = lu.decomp(N.tolist())
            
            deltap = lu.solve(LU, permut, grad)
            
            p1 = p0 + numpy.array(deltap)
            
            # Calculate the goal function
            tmpgoal = 0.0
            for i in range(len(temps)):
                
                residuals[i,0] = temps[i] - climatesignal.abrupt(depths[i], \
                                                                 p1[0], p1[1])
                
                tmpgoal += residuals[i,0]**2
                        
                                                
            if tmpgoal >= goal[iteration-1]:
                
                marq_param *= 10
                
            else:
                
                marq_param *= 0.1
                
                break        
                
        goal.append(tmpgoal)                
        end = clock()
        sys.stderr.write(" Done (%g s)\n" % (end-start))            
        sys.stderr.write("  Marquardt iterations = %d\n" % \
                         (marq_it))
        sys.stderr.write("  Goal function = %g\n" % (goal[iteration]))
        sys.stderr.write("  Solution: mag=%g   time=%g\n" % (p1[0], p1[1]))
        
        # Get out of the iterations if there is stagnation
        if abs(goal[iteration] - goal[iteration-1]) < 10**(-8):
                        
            break
        
    # Print the results
    totalend = clock()
    sys.stderr.write("\nFinished (%g s)!\n" % (totalend-totalstart))
    sys.stderr.write("\nResults:\n")
    sys.stderr.write("  iterations: %d\n" % (iteration))
    sys.stderr.write("  Goal function: %g\n" % (goal[iteration]))
    sys.stderr.write("  Magnitude: %g\n" % (p1[0]))
    sys.stderr.write("  Time: %g\n" % (p1[1]))
    sys.stdout.write("\nResults:\n")
    sys.stdout.write("  iterations: %d\n" % (iteration))
    sys.stdout.write("  Goal function: %g\n" % (goal[iteration]))
    sys.stdout.write("  Magnitude: %g\n" % (p1[0]))
    sys.stdout.write("  Time: %g\n" % (p1[1]))
    
    pylab.figure()
    pylab.hist(residuals.T.tolist()[0], \
               bins=len(residuals)/8, \
               normed=False, facecolor='gray', alpha=0.75)
    pylab.title('Residuos')
    pylab.ylabel('Contagem')
    pylab.xlabel('Residuos (C)')
    
    adjusted_temps = []
    for i in range(len(depths)):        
        adjusted_temps.append(climatesignal.abrupt(depths[i], p1[0], p1[1]))
    
    pylab.figure()
    pylab.plot(temps, -1*depths, '.k', label='Medido')
    pylab.plot(adjusted_temps, -1*depths, '.-r', label='Ajustado')
    pylab.legend(numpoints=1, prop={'size':7})
    pylab.title('Solucao Final')
    pylab.xlabel('Temperatura (C)')
    pylab.ylabel('Profundidade (m)')
    
           
    pylab.figure()
    pylab.plot(goal, '.-k')
    pylab.title('Funcao Objetivo')
    pylab.xlabel('Iteracao')
    pylab.ylabel('Erro')
    pylab.show()
    
    
if __name__ == '__main__':
    
    
    # Make a synthetic model
    ############################################################################
    magnitude = 2
    time = 100
    startdepth = 10
    maxdepth = 100
    dz = 1
    sys.stderr.write("Synthetic model:\n")
    sys.stderr.write("  magnitude: %g C\n" % (magnitude))
    sys.stderr.write("  time: %g years\n" % (time))
    depths = numpy.arange(startdepth, maxdepth+dz, dz)
    
    temps = []
    
    for z in depths:
        
        temps.append(climatesignal.abrupt(z, magnitude, time))
#        temps.append(climatesignal.linear(z, magnitude, time))
        
    ############################################################################

    # Or read from a file
    ############################################################################
#    import sys
#    depths, temps = pylab.loadtxt(sys.argv[1], unpack=True)
    ############################################################################

        
    invert_abrupt(temps, depths, mag0=1, t0=1, \
                  reg_param=0.0000001, marq_param=0.1, contam=True)
#    invert_linear(temps, depths, mag0=0.1, t0=10, \
#                  reg_param=0.00000001, marq_param=0.1, contam=True)