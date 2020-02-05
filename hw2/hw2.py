import numpy as np
from numpy import ndarray

''' One optimum at x*=(0,0), f*=0. '''
def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

''' One optimum at x*=(1,1), f*=0. '''
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2

def brachis(x):
    pass

def myfunc(x):
    func = matyas
    # func = rosenbrock
    # func = brachis
    
    f = func(x)
    g = 


def uncon(func, x0, epsilon_g, options=None):
    if options is None: 
        #Set up default options
        debug = 1

    optimizer = OptimizerUncon(func, x0, epsilon_g)
    # optimizer.forward_difference(x0)
        
    return x_opt, f_opt, outputs

class OptimizerUncon:
    def __init__(self, func, x0, epsilon_g, options=None):
        self.alpha = 1
        self.eps_g = epsilon_g
        self.x0 = x0
        self.n = len(x0)
        self.func = func
        self.h = 1e-8
        self.iterations = 0
        
        if options is None:
            self.gradient = self.forward_difference
    
    def objective(self, x):
        f, g = self.func(x)
        self.iterations += 1
        return f, g
        
    def gradient(self, x:ndarray, func:function=None) -> (ndarray, ndarray):
        return np.array([-1]), np.array([-1]) 
        
    def forward_difference(self, x:ndarray, func:function=None) -> (ndarray, ndarray):
        if func is None:
           func = self.func 

        n = self.n
        f = func(x)
        g = np.zeros(n)
        for j in range(n):
            e = np.zeros(n)
            e[j] = self.h
            # forward differencing
            g[j] = (func(x+e) - f))/self.h
            
        return f, g
    
    def quasi_newton(self, x:ndarray) -> 
        ''' pk = -Vk * gk  --  for first iteration, -V = -I (identity)
            Vk1*(gk1 - gk) = xk1 - xk
            yk = gk1 - gk
        '''

        n = self.n
        V = np.eye(n)
        g = self.gradient(x)
        p = -V @ g


if __name__ == '__main__':
    x0 = np.array([2, 3])
    epsilon_g = 1e-6
    options
    x_opt, f_opt, outputs = uncon(myfunc, x0, epsilon_g, options)

    print(f'x_opt: {x_opt}')
    print(f'f_opt: {f_opt}')
    print(f'outputs: {outputs}')
    
