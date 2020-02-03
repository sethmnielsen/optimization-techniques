import numpy as np 

'''
Parameters:
func: function handle to a function of the form f, g = func(x) where f
        is the function value and g is a column vector containing the gradients.
        x is a design variable.
x0: 1D array containing the initial guess
epsilon_g: float: convergence tolerance. termination occurs when the inifinity
            norm of the gradient is less than epsilon_g
options: dictionary a dictionary containing the options. You can use this to 
            try different algorithm choices. I will pass nothing in so check 
            if it is None and set up some defaults
Outputs:
xopt : 1D array with the optimal solution
fopt : float containing the objective function value
outputs: Dictionary: other miscelaneous outputs that you might want. For example
        an array containing the convergence metric at each iteration.
'''
def uncon(func, x0, epsilon_g, options=None):
    if options is None: 
        debug = 1
        #Set up default options
    
    '''
    Your code goes here. You should call other functions but do not change this
    function signature. This is the file I will call to test your algorithm.
    '''

    return x_opt, f_opt, outputs
    
    
    



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