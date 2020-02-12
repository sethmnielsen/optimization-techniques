import numpy as np
from numpy import ndarray
import functools
from hw2_plotter import Plotter

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from seaborn import xkcd_rgb as xcolor

np.set_printoptions(floatmode='unique')


func_evals = 0

# region
def my_custom_test_func(test_func):
    @functools.wraps(test_func)
    def wrapper_my_custom(x):
        f = test_func(x)
        g = OptimizerUncon.gradient(test_func, x, f)
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
def brachis(y_dv):
    h = 1.0 # initial y
    mu = 0.3 # coeff of friction
    n = 60

    x_arr = np.linspace(0, 1, n)  # fixed
    y = np.hstack((1.0, y_dv, 0.0))

    y_low_lim = -0.9*x_arr+0.9
    y_high_lim = np.flip(x_arr)

    # mask_low = y<y_low_lim
    # if np.any(mask_low):
    #     # print(f'\ny_low: {y[mask_low]}')
    #     # print(f'low_inds: {np.nonzero(mask_low)[0]}')
    #     y[mask_low] = y_low_lim[mask_low]

    # mask_high = y>y_high_lim
    # if np.any(mask_high):
    #     # print(f'\ny_high: {y[mask_high]}')
    #     # print(f'high_inds: {np.nonzero(mask_high)[0]}')
    #     y[mask_high] = y_high_lim[mask_high]

    time_sum = 0
    for i in range(n-1):
        # Loop over x and y arrays
        xi = x_arr[i]
        xip = x_arr[i+1]

        yi = y[i]
        yip = y[i+1]
        dx = xip-xi
        dy = yip-yi

        # Gravity not needed - will multiply it for final result
        # a = np.sqrt(2.0/g)
        b = np.sqrt(dx**2+dy**2)
        q = h-yip
        r = mu*xip
        s = q-r
        if s < 0:
            print('negaaatttiiiivvvveee    s')
        t = h-yi
        u = mu*xi
        v = t-u
        if v < 0:
            print('negatiiiivivvveieieie     v')
        c = np.sqrt(s) + np.sqrt(v)

        time_sum += b/c

    return time_sum * np.sqrt(2/9.81) # multiply gravity

# endregion


""" THIS IS THE FUNCTION THAT WILL ACTUALLY BE CALLED BY D NING """
def uncon(func, x0, epsilon_g, options=None):
    if options is None:
        # set defaults here for how you want me to run it.
        options = {'pfunc': 'steepest_descent',
                   'afunc': 'line_search',
                   'debug': False,
                   'plot_x_vec': False}


    optimizer = OptimizerUncon(func, x0, epsilon_g, options)
    if options['plot_x_vec']:
        x_plot = Plotter()
        fig, axes = x_plot.make_new_plot(1,1,'', ['x1'], ['x2'])
        ax1 = axes[0]
        dx = 0.1
        x_vals = np.arange(-10.1,10.1,dx)
        x1, x2 = np.meshgrid(x_vals,x_vals)

        x3d = np.array([x1,x2])
        z, _ = func(x3d)
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
        cf = ax1.contourf(x1+dx/2, x2+dx/2, z, levels)
        # cf = ax1.contourf(x1+dx/2, x2+dx/2, z, locator=LogLocator(base=2.0))
        fig.colorbar(cf, ax=ax1)
        ax1.set_title('rosenbrock contour plot')

        plt.show()
        return None, None, None
    else:
        try:
            sol = optimizer.minimize()
        except KeyboardInterrupt:
            sol = optimizer.finish(False)
        x_opt = sol.x_opt
        f_opt = sol.f_opt
        outputs = sol.outputs

    return x_opt, f_opt, outputs

class OptimizerUncon:
    def __init__(self, func, x0, epsilon_g, options):
        self.func = func
        self.eps_g = epsilon_g
        self.options = options

        self.init = False

        self.outputs = ['dummy value']
        n = len(x0)

        self.n = n
        self.V:ndarray = np.eye(n)

        self.x = np.copy(x0)
        self.x_prev = np.copy(x0)
        self.f = 0.
        self.g = np.zeros(n)
        self.g_prev = np.zeros(n)
        self.p = np.zeros(n)
        self.plotter = Plotter()

        self.iterations = 0
        self.iters_arr = []
        self.gmax_arr = []

        # Tunables
        self.alpha_guess = 0.1
        self.alpha = self.alpha_guess
        self.alpha_max = 0.4
        self.rho = 0.7
        self.mu1 = 1e-4
        self.mu2 = 0.5
        self.iters_limit = 5.0e5

    @staticmethod
    def gradient(func, x:ndarray, f:float, h=1e-8, run=True) -> (ndarray):
        n = len(x)
        if not run:
            return np.zeros(n)
        g = np.zeros(n)
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            # forward differencing
            g[j] = (func(x+e) - f)/h
        return g


    def minimize(self):
        self.create_plot('Convergence Plot', 'Iterations', r'$\Vert{g}\Vert$')
        self.f, self.g = self.func(self.x)
        while self.iterations < self.iters_limit:

            self.p = self.choose_search_dir()
            self.alpha = self.choose_step_size()
            max_g = np.abs(np.max(self.g))

            if self.iterations % 50 == 0:
                self.redraw(max_g)

            self.iterations += 1

            if max_g < self.eps_g:
                self.redraw(max_g)
                return self.finish()

            if self.iterations % 50 == 0 and self.options['debug']:
                print('------- STATS -------')
                print(f'current iter: {self.iterations}')
                print(f'current x: {self.x}')
                print(f'current f: {self.f}')
                print(f'current a: {self.alpha}')
                print(f'current g: {self.g}')
                print(f'   max(g): {max_g}')
                print(f'   diff_g: {self.eps_g - max_g}\n')

        return self.finish(False)

    def redraw(self, max_g):
        self.iters_arr.append(self.iterations)
        self.gmax_arr.append(max_g)
        self.plotter.update_plot(0,0,0,self.iters_arr, self.gmax_arr)


    def create_plot(self, title, xlabs, ylabs):
        title = 'Convergence Plot'
        xlabs = ['Iterations']
        ylabs = [r'$\Vert{g}\Vert$']

        self.plotter.make_new_plot(1, 1, title, xlabs, ylabs)
        self.plotter.init_xy_data(0, 0, [], [], '')
        self.plotter.axes[0][0].set_yscale("log")
        self.plotter.axes[0][0].axhline(y=self.eps_g,ls='--',c=xcolor['pale red'], label=r'$\tau$')


    def finish(self, converged=True):
        sol = Solution(self.x, self.f, self.outputs)

        if converged:
            print("--------- OPTIMIZATION COMPLETE ------------")
            iterations_report = f"{self.iterations}"
        elif not converged:
            print("--------- OPTIMIZATION COMPLETE - DID NOT CONVERGE ------------")
            if self.iterations == self.iters_limit:
                info = "(maximum allowed)"
            else:
                info = "(keyboard interrupt)"
            iterations_report = f"{self.iterations} {info}"

        print(f'x_opt: {sol.x_opt}')
        print(f'f_opt: {sol.f_opt}')
        print(f'final step size: {self.alpha}')
        print(f'outputs: {sol.outputs}\n')
        print(f'num of iterations: {iterations_report}')
        if self.options['debug']:
            print(f'num of func evals: {func_evals}')
            print(f'final gradient: {self.g}')
            print(f'final gmax: {np.max(self.g)}')
        self.plotter.hold_plot()
        self.plotter.close_plots()
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
            self.init = True
            return -1*self.V @ self.g

        I = np.eye(self.n)
        s = self.alpha * self.p
        y = self.g - self.g_prev
        den = np.outer(s,y)
        divmat = (s @ y)/den

        self.V = (I - divmat) @ self.V @ (I - divmat) + (s@s)/den

        p = -1*self.V @ self.g
        return p

    def line_search(self):
        # phi(alpha) = phi(0) +  mu1*alpha*phi'(0)
        # phi(alpha) =    f   +  mu1*alpha* gT*p
        # f, mu1, alpha, g, p = self.f, self.mu1, self.alpha, self.g, self.p
        phi0 = self.f  # first phi has alpha=0, so phi(0) = f(xk)
        phi = phi0
        rhs = phi0 + self.mu1*alpha*(g @ self.p)
        alpha = 0
        x = np.zeros(self.n)
        cnt = 0
        while phi > rhs:
            if cnt == 0:
                alpha = self.alpha
                alpha = self.alpha_guess
            else:
                alpha *= self.rho
            xk1 = self.x + alpha*self.p
            f, g = self.func(xk1)
            phi = f
            rhs = phi0 + self.mu1*alpha*(g @ self.p)
            cnt += 1
        self.g_prev = np.array(self.g)
        self.g = np.array(g)
        self.f = f
        self.x_prev = np.copy(self.x)
        self.x = xk1
        return alpha

    def bracketed_ls(self):
        phi0 = self.f
        self.phi_prime0 = self.g @ self.p  # phi_prime at current xk
        phi_prev = self.f
        alpha_prev = 0.
        # alpha = self.alpha
        alpha = self.alpha_guess
        alpha_max = self.alpha_max

        g_prev = np.array(self.g)
        g = np.zeros(self.g.shape)
        i = 0
        while True:
            if i > 100:
                alpha_star = np.random.uniform(alpha, alpha_max)
                break
            x_new = self.x + alpha*self.p
            f, g = self.func(x_new)
            phi = f
            rhs = phi0 + self.mu1*alpha*g @ self.p
            if (phi > rhs) or (phi > phi_prev and i > 0):
                alpha_star, x_new, f, g = self.pinpoint(g_prev, alpha_prev, phi_prev, alpha, phi)
                break
            phi_prime = g @ self.p
            if abs(phi_prime) <= -self.mu2*self.phi_prime0:
                alpha_star = alpha
                break
            elif phi_prime >= 0:
                alpha_star, x_new, f, g = self.pinpoint(g, alpha, phi, alpha_prev, phi_prev)
                break
            else:
                alpha_next = np.random.uniform(alpha, alpha_max)
                alpha_prev = alpha
                alpha = alpha_next
                phi_prev = phi
                i += 1
        self.g_prev = np.array(self.g)
        self.g = g
        self.f = f
        self.x = x_new
        return alpha_star

    def pinpoint(self, gl, a_low, phi_low, a_high, phi_high) -> float:
        g_new = np.zeros(gl.shape)
        f_new = 0
        alpha_star = 0
        alpha_new = self.alpha_guess
        phi0 = self.f
        phi_prime0 = self.phi_prime0
        phi_prime_low = gl @ self.p  # phi_low, gl = func(x + a_low*p)
        j = 0
        while True:
            if j > 1000:
                alpha_star = alpha_new
                break
            # interpolate alpha_new
            numer = 2*a_low*(phi_high - phi_low) + phi_prime_low*(a_low**2 - a_high**2)
            denom = 2*( phi_high - phi_low + phi_prime_low*(a_low - a_high) )
            alpha_new = numer / denom
            # Evaluate new objective, gradient (f, g)
            x_new = self.x + alpha_new*self.p
            f_new, g_new = self.func(x_new)
            phi_new = f_new
            if (phi_new > phi0 + self.mu1*alpha_new*phi_prime0) or (phi_new > phi_low):
                a_high = alpha_new
                phi_high = phi_new
            else:
                phi_prime_new = g_new @ self.p
                if abs(phi_prime_new) <= -self.mu2*phi_prime0:
                    alpha_star = alpha_new
                    break
                elif phi_prime_new*(a_high-a_low) >= 0:
                    a_high = a_low
                    phi_high = phi_low
                a_low = alpha_new
                gl = np.array(g_new)
                phi_low = phi_new
                phi_prime_low = phi_prime_new
            j += 1
        return alpha_star, x_new, f_new, g_new


class Solution:
    def __init__(self, x_opt, f_opt, outputs):
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.outputs = outputs


if __name__ == '__main__':
    """ THIS IS ME SIMULATING WHAT D NING WILL RUN ON HIS SIDE """

    # EXCEPT FOR THIS PART; NING WILL NOT PASS OPTIONS
                # 'afunc': 'line_search',
    options =  {'pfunc': 'quasi_newton',
                'afunc': 'line_search',
                'debug': True,
                'plot_x_vec': False}


    epsilon_g = 1e-5
    myfunc = matyas
    # myfunc = rosenbrock
    # myfunc = brachis

    if myfunc == brachis:
        x0 = np.linspace(1.0,0.0,60)[1:-1]
    else:
        x0 = np.array([2, 3])

    x_opt, f_opt, outputs = uncon(myfunc, x0, epsilon_g, options)
