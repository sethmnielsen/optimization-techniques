import numpy as np
import pyoptsparse as pyop

# Solve the Brachistochrone Problem
# Find the min time between 2 pts for a particle subject to gravity only

class Minimize:
    def __init__(self):
        # Create optimization problem
        self.op: pyop.Optimization = pyop.Optimization('brachistochrone', self.objfunc)
        self.mu = 0.3

        num_pts = 10  # 12 pts including start and end

        xvals = np.linspace(0, 1, num_pts)
        yvals = np.linspace(1, 0, num_pts)
        self.op.addVarGroup('y', nVars=num_pts, type='c', value=yvals, lower=None, upper=None)
    
    def objfunc(self):
         
        np.sqrt()