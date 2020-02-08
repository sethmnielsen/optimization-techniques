import numpy as np
from numpy import ndarray
import functools
np.set_printoptions(floatmode='unique')


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


""" THIS IS THE FUNCTION THAT WILL ACTUALLY BE CALLED BY D NING """
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

        self.init = False

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
        self.iters_limit = 5.0e5

    def minimize(self):
        while self.iterations < self.iters_limit:
            self.f, self.g = self.func(self.x)

            self.p = self.choose_search_dir() # done
            self.choose_step_size()           # done
            max_g = np.max(self.g)
            if max_g < self.eps_g:
                return self.finish()

            self.iterations += 1

            if self.iterations % 10000 == 0:
                print('------- STATS -------')
                print(f'current iter: {self.iterations}')
                print(f'current x: {self.x}')
                print(f'current f: {self.f}')
                print(f'current a: {self.alpha}')
                print(f'current g: {self.g}')
                print(f'   max(g): {max_g}')
                print(f'   diff_g: {self.eps_g - max_g}\n')

        return self.finish(False)


    def finish(self, converged=True):
        sol = Solution(self.x, self.f, self.outputs)

        if converged:
            print("--------- OPTIMIZATION COMPLETE ------------")
            iterations_report = f"{self.iterations}"
        elif not converged:
            print("--------- OPTIMIZATION COMPLETE - DID NOT CONVERGE ------------")
            iterations_report = f"{self.iterations} (maximum allowed)"

        print(f'x_opt: {sol.x_opt}')
        print(f'f_opt: {sol.f_opt}')
        print(f'final step size: {self.alpha}')
        print(f'outputs: {sol.outputs}\n')
        print(f'num of iterations: {iterations_report}')
        if self.options['debug']:
            print(f'num of func evals: {func_evals}')
        return sol

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

    def quasi_newton(self):
        ''' pk = -Vk * gk  --  for first iteration, -V = -I (identity)
            Vk1*(gk1 - gk) = xk1 - xk
            yk = gk1 - gk
        '''
        if not self.init:
            self.V = np.eye(self.n)
        _, self.g = self.func(self.x)
        p = -self.V @ self.g
        return p

    def line_search(self):
        # phi(alpha) = phi(0) +  mu1*alpha*phi'(0)
        # phi(alpha) =    f   +  mu1*alpha* gT*p
        # f, mu1, alpha, g, p = self.f, self.mu1, self.alpha, self.g, self.p
        phi0 = self.f  # first phi has alpha=0, so phi(0) = f(xk)
        phi = phi0
        # backtrack lambda : phi0 + self.mu1*self.alpha * self.g @ self.p
        rhs = 0
        alpha = 0
        x = np.zeros(self.n)
        cnt = 0
        while phi > rhs:
            if cnt == 0:
                alpha = self.alpha
            else:
                alpha *= self.rho
            xk1 = self.x + alpha*self.p
            f, g = self.func(xk1)
            phi = f
            rhs = phi0 + self.mu1*alpha*g @ self.p
            cnt += 1
        self.alpha = alpha
        self.g = g
        self.f = f
        self.x = xk1

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
    """ THIS IS ME SIMULATING WHAT D NING WILL RUN ON HIS SIDE """

    # EXCEPT FOR THIS PART; NING WILL NOT PASS OPTIONS
    options =  {'pfunc': 'steepest_descent',
                'afunc': 'line_search',
                'debug': True}

    x0 = np.array([2, 3])
    epsilon_g = 1e-5
    # myfunc = matyas
    myfunc = rosenbrock
    x_opt, f_opt, outputs = uncon(myfunc, x0, epsilon_g, options)

    # print(f'x_opt: {x_opt}')
    # print(f'f_opt: {f_opt}')
    # print(f'outputs: {outputs}')
