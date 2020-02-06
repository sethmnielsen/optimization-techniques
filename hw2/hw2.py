import numpy as np
from numpy import ndarray
import functools

func_evals = 0

def my_custom_test_func(test_func):
    @functools.wraps(test_func)
    def wrapper_my_custom(x):
        f = test_func(x)
        g = OptimizerUncon.gradient(x, f, h=1e-8, func=test_func)
        return f, g
    return wrapper_my_custom
            
def func_evals_counter(test_func):
    @functools.wraps(test_func)
    def wrapper_fevals_counter(x):
        global func_evals
        func_evals += 1
        return test_func(x)
    return wrapper_fevals_counter


''' One optimum at x*=(0,0), f*=0. '''
@my_custom_test_func
@func_evals_counter
def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

''' One optimum at x*=(1,1), f*=0. '''
@my_custom_test_func
@func_evals_counter
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2

@my_custom_test_func
@func_evals_counter
def brachis(x):
    return 0.


""" THIS IS THE FUNCTION THAT WILL ACTUALLY BE CALLED BY NING """
def uncon(func, x0, epsilon_g, options=None):
    if options is None: 
        # set defaults here for how you want me to run it.
        options = {'pfunc': 'steepest_descent',
                   'afunc': 'line_search',
                   'debug': False}
        

    optimizer = OptimizerUncon(func, x0, epsilon_g, options)
    sol = optimizer.minimize()
    x_opt = sol.x_opt
    f_opt = sol.f_opt
    outputs = sol.outputs
        
    return x_opt, f_opt, outputs

class OptimizerUncon:
    def __init__(self, func, x0, epsilon_g, options):
        self.func = func
        self.x0 = x0
        self.eps_g = epsilon_g
        self.options = options
        
        self.outputs = ['dummy value']
        n = len(x0)

        self.n = n
        self.V = np.eye(n)

        self.x = np.copy(x0)
        self.f = 0.
        self.g = np.zeros(n)
        self.p = np.zeros(n)
       
        self.iterations = 0
        
        # Tunables 
        self.alpha = 1
        self.h = 1e-8
        self.rho = 0.5
        self.mu1 = 1e-4
        self.iters_limit = 1000
        
    def minimize(self):
        while self.iterations < self.iters_limit:
            self.f, self.g = self.func(self.x)

            self.p = self.choose_search_dir() # done
            self.choose_step_size()           # done
            if self.update() < self.eps_g:
                sol = Solution(self.x, self.f, self.outputs)
                
                print("--------- OPTIMIZATION COMPLETE ------------")
                print(f'x_opt: {sol.x_opt}')
                print(f'f_opt: {sol.f_opt}')
                print(f'outputs: {sol.outputs}\n')
                print(f'num of iterations: {self.iterations}')
                if self.options['debug']:
                    print(f'num of func evals: {func_evals}')
                return sol 
            
            self.iterations += 1

    def choose_search_dir(self) -> ndarray:
        pfunc = self.options['pfunc']

        if pfunc == 'steepest_descent':
            p = self.steepest_descent()
        elif pfunc == 'quasi_newton':
            p = self.quasi_newton()

        p_unit = p / np.linalg.norm(p)
        return p_unit

    def choose_step_size(self) -> float:
        afunc = self.options['afunc']
        if afunc == 'line_search':
            return self.line_search()
        elif afunc == 'bracketed_ls':
            return self.bracketed_ls()
    
    def steepest_descent(self) -> ndarray:
        return -self.g
    
    def quasi_newton(self, x:ndarray):
        ''' pk = -Vk * gk  --  for first iteration, -V = -I (identity)
            Vk1*(gk1 - gk) = xk1 - xk
            yk = gk1 - gk
        '''
        if not self.init:
            self.V = np.eye(self.n)
        self.g = self.gradient(x)
        p = -self.V @ self.g

    def line_search(self):
        # phi(alpha) = phi(0) +  mu1*alpha*phi'(0)
        # phi(alpha) =    f   +  mu1*alpha* gT*p
        # f, mu1, alpha, g, p = self.f, self.mu1, self.alpha, self.g, self.p
        phi = self.f 
        while  
        phi = self.f + self.mu1*self.alpha * self.g @ self.p
                
    def bracketed_ls(self):
        pass

    @staticmethod
    def gradient(x:ndarray, f:float, h:float, func) -> (ndarray):
        n = len(x)
        g = np.zeros(n)
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            # forward differencing
            g[j] = (func(x+e) - f)/h
            
        return g
    
class Solution:
    def __init__(self, x_opt, f_opt, outputs):
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.outputs = outputs


if __name__ == '__main__':
    """ THIS IS ME SIMULATING WHAT NING WILL RUN ON HIS SIDE """


    # EXCEPT FOR THIS PART; NING WILL NOT PASS OPTIONS
    options =  {'pfunc': 'steepest_descent',
                'afunc': 'line_search',
                'debug': True}
    
    x0 = np.array([2, 3])
    epsilon_g = 1e-6
    myfunc = matyas 
    x_opt, f_opt, outputs = uncon(myfunc, x0, epsilon_g, options)

    print(f'x_opt: {x_opt}')
    print(f'f_opt: {f_opt}')
    print(f'outputs: {outputs}')
