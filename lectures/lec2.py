import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def J(x, *args): 
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def cfuncs(x):
    return np.array([ x[0]**2 + x[1]**2, 
                      x[0] + 3*x[1] ])

bounds = np.array([[-np.inf, -np.inf],
                   [      1,       5]])

constrs = NonlinearConstraint(cfuncs, bounds[0], bounds[1])
guess = np.array([100,100])

res = minimize(J, guess, method='SLSQP', constraints=constrs, options={'disp': True})
# res = minimize(J, np.array([0,0]), method='BFGS', options={'disp': True})

print(f'\n{res}')
