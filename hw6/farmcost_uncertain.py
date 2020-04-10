import numpy as np
import matplotlib.pyplot as plt
import pyoptsparse as pyop

def jensenwake(x,y,alpha,beta,delta):
    if x < 0: return 1.0

    theta = np.arctan2(y,(x+delta))
    f = 0.0
    if np.abs(theta) < np.pi/beta:
        f = 0.5*(1+np.cos(beta*theta))

    # normalized by Uinf
    V = 1 - 2/3*(0.5/(0.5+alpha*x))**2 * f
    return V

def farmcost(yback, alpha, beta, delta, plotit=False):
    x = np.array([0.0, 4.0, 8.0])
    y = np.hstack([0, yback]) # append 0 in front for lead turbine
    CP = 0.38
    rho = 1.2

    u12 = jensenwake(x[1]-x[0], y[1]-y[0], alpha, beta, delta)
    u13 = jensenwake(x[2]-x[0], y[2]-y[0], alpha, beta, delta)
    u23 = jensenwake(x[2]-x[1], y[2]-y[1], alpha, beta, delta)

    u = np.array([1.0, u12, 1-np.sqrt((1-u13)**2 + (1-u23)**2)])
    P = CP*0.5*rho*u**3*np.pi*1.0**2/4

    area = np.mean((x - np.mean(x))**2 + (y - np.mean(y))**2)
    othercost = 1.0

    power = np.sum(P)

    coe = (area/100 + othercost)/power

    if plotit:
        nx = 200
        ny = 250

        xvec = np.linspace(-2,12,nx)
        yvec = np.linspace(-5,10,ny)

        Y,X = np.meshgrid(yvec, xvec)
        U1 = np.zeros([nx,ny])
        U2 = np.zeros([nx,ny])
        U3 = np.zeros([nx,ny])

        for i in range(nx):
            for j in range(ny):
                U1[i,j] = jensenwake(X[i,j]-x[0], Y[i,j]-y[0], alpha,beta,delta)
                U2[i,j] = jensenwake(X[i,j]-x[1], Y[i,j]-y[1], alpha,beta,delta)
                U3[i,j] = jensenwake(X[i,j]-x[2], Y[i,j]-y[2], alpha,beta,delta)

        Udef = 1 - np.sqrt((1-U1)**2 + (1-U2)**2 + (1-U3)**2)

        plt.figure()
        plt.contourf(X, Y, Udef, 200)
        plt.colorbar()
        plt.xlabel('x/D')
        plt.ylabel('y/D')

        plt.show()
    
    return coe

def obj_farmcost(data, plotit=False):
    alpha = 0.1
    beta = 9
    delta = 5
    
    y_arr = data['y']
    
    funcs = {}
    
    funcs['coe'] = farmcost(y_arr, alpha, beta, delta, plotit)
    funcs['y3y2'] = constraint(*y_arr)
    fail = False
    
    return funcs, fail

def constraint(y2, y3):
    return y3 - y2


def optimize_farmcost(*args):
    y0 = np.array([1, 10])
    
    funcs, _ = obj_farmcost({"y": y0})
    print(f'initial COE: {funcs["coe"]}')
    # starting point and bounds

    # objective
    opt_prob: pyop.Optimization = pyop.Optimization('coe', obj_farmcost)
    
    # linear constraint
    # the constraint y3 > y2 can be formulated as a linear constraint.
    opt_prob.addVarGroup('y', nVars=2, value=y0, lower=0, upper=10) 
    opt_prob.addConGroup('y3y2', nCon=1, lower=1e-5) 
    
    opt_prob.addObj('coe')
    
    optimizer = pyop.SNOPT()
    
    # deterministic optimization
    sol: pyop.pyOpt_solution.Solution = optimizer(opt_prob, sens='FD')
    
    print(f'{sol.fStar.item() = }')
    print(f'{sol.xStar["y"] = }')
    
    obj_farmcost(sol.xStar, plotit=False)

def monte_carlo():
    # try increasing large samples
    nvec = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    yback = np.array([1, 10])

    # initialize the array of results from each sample size
    cost = np.zeros(len(nvec))

    for i in range(len(nvec)):  # repeat Monte Carlo for many different sample sizes
        n = nvec[i]

        # TODO: generate random numbers from the input distributions.
        
        
        # initialize for Monte Carlo
        costvec = np.zeros(n)
        
        for j in range(n):  # loop over all of the trials
            costvec[j] = farmcost(yback, randa[j], randb[j], randd[j])
        
        # TODO: compute some statistic using costvec (e.g., mean, prctile, etc.)
        cost[i] = 

    figure
    semilogx(nvec, cost, 'o')
        

if __name__ == '__main__':
    optimize_farmcost()