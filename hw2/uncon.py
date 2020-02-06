import numpy as np 
from hw2.hw2 import *

"""An algorithm for unconstrained optimization.

Parameters
----------
func : function handle
        function handle to a function of the form: f, g = func(x)
        where f is the function value and g is a numpy array containing
        the gradient. x are design variables only.
x0 : ndarray
        starting point
epsilon_g : float
        convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
options : dict
        a dictionary containing options.  You can use this to try out different
        algorithm choices.  I will not pass anything in, so if the input is None
        you should setup some defaults.

Outputs
-------
xopt : ndarray
        the optimal solution
fopt : float
        the corresponding function value
outputs : list
        other miscelaneous outputs that you might want, for example an array
        containing a convergence metric at each iteration.
"""


def uncon(func, x0, epsilon_g, options=None):
    if options is None: 
        # set defaults here for how you want me to run it.

    """ Your code goes here!  You can (and should) call other functions, but make
        sure you do not change the function signature for this file.  This is the
        file I will call to test your algorithm.
    """
        
    return x_opt, f_opt, outputs