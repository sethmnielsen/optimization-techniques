import numpy as np
from scipy.optimize import minimize

# ------ unconstrained -----------
# starting point
x0 = [0.0, 0.0]

# objective (rosenbrock)
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# set options
options = {'disp': True}

# call minimize
res = minimize(rosenbrock, x0, method='BFGS', options=options)
print('xopt =', res.x)
print('fopt =', res.fun)

# ------ constrained -----------

# starting point, lower/upper bounds
x0 = [0.0, 0.0]
bounds = [(-5.0, 5.0), (-5.0, 5.0)]

# set options
options = {'disp': True}

# constraint function
def con(x):
    # convention for minimize is c > 0
    c = np.zeros(2)
    c[0] = -x[0]**2 - x[1]**2 + 1
    c[1] = -x[0] - 3*x[1] + 5
    return c

constr = {'type': 'ineq', 'fun': con}

# call minimize
res = minimize(rosenbrock, x0, method='SLSQP', constraints=constr, options=options)
print('xopt =', res.x)
print('fopt =', res.fun)