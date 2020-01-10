import numpy as np
from scipy import optimize as op
from scipy.optimize import NonlinearConstraint

def J(x, *args): 
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def cf1(x):
    return np.array([ x[0]**2 + x[1]**2 ])

def cf2(x):
    return x[0] + 3*x[1]

def cfuncs(x):
    return np.array([ x[0]**2 + x[1]**2, 
                      x[0] + 3*x[1] ])

c1 = NonlinearConstraint(cf1, -np.inf, 1)
c2 = NonlinearConstraint(cf2, -np.inf, 5)

bounds = np.array([[-np.inf, -np.inf],
                          1,       5])

constrs = NonlinearConstraint(cfuncs, bounds[0], bounds[1])
guess = np.array([100,100])

# res = op.minimize(J, guess, method='SLSQP', constraints=[c1, c2], options={'disp': True})
res = op.minimize(J, guess, method='SLSQP', constraints=constrs, options={'disp': True})
# res = op.minimize(J, np.array([0,0]), method='BFGS', options={'disp': True})

print(f'\n{res}')