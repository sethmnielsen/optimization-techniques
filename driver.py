import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=7)

function_evals = 0
number_of_points = 60
convergences = np.array([[0,0]])

def matyas(x):
    global function_evals
    function_evals += 1
    return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]

def rosenbrock(x):
    global function_evals
    function_evals += 1
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def brachistochrone(in_y):
    global function_evals
    function_evals += 1
    H = 1
    W = 1
    mu_k = 0.3
    n = number_of_points

    y_end_points = np.append(in_y,0)
    y_start_points = np.append(H,in_y)
    del_y_vec = y_start_points - y_end_points

    x = np.linspace(0,W,n)
    x_end_points = x[1:]
    x_start_points = x[:-1]
    del_x_vec = x_start_points - x_end_points
    return np.sum((np.sqrt(np.abs(del_x_vec**2+del_y_vec**2)))/(np.sqrt(np.abs(H-y_end_points-mu_k*x_end_points))+np.sqrt(np.abs(H-y_start_points-mu_k*x_start_points))))

def approx_gradient_black_box(func_in, x_in):
    h = 1e-8
    f0 = func_in(x_in)
    out = np.zeros(len(x_in))
    # print(x)
    for i in range(len(x_in)):
        # print(i)
        x_in[i] += h
        # print(x)
        out[i] = (func_in(x_in)-f0)/h
        # print(out[i])
        x_in[i] -= h
    return out

def complex_gradient_calc(func_in, x_in):
    h = 1e-15
    f0 = func_in(x_in)
    out = np.zeros(len(x_in))
    x_copy = np.copy(x_in)
    for i in range(len(x_copy)):
        x_new = x_copy.astype(complex)
        x_new[i] += h*1j
        out[i] = np.imag(func_in(x_new)-f0)/h
    return out

def find_gradient(func_in, x_in):
    return approx_gradient_black_box(func_in, x_in)
    # return complex_gradient_calc(func_in, x_in)

def find_search_direction_steepest_decent(func_in, x_in):
    grad = find_gradient(func_in, x_in)
    return grad/-np.linalg.norm(grad)

def find_search_direction_quasi_newton(func_in, x_in):
    pass

def line_search_naive(func, x, g, p):
    mu_1 = 1e-4
    alpha = 1
    rho = 0.5
    f0 = func(x)
    out = f0 + mu_1*alpha*g.T@p
    while out > f0:
        alpha = alpha*rho
        out = f0 + mu_1*alpha*g.T@p
    return alpha

def line_search_better(func_in, x_in, g, p, alpha_guess, alpha_max):
    mu_1 = 1e-4
    mu_2 = 0.5 #this should be between 0.1 and 0.9
    phi_0 = func_in(x_in)
    alpha_prev = 0.0
    alpha = alpha_guess
    phi_prime_0 = g.T@p
    phi_alpha_prev = func_in(x_in + alpha_prev*p)
    i = 1
    loop_calls = 0
    max_loop_calls = 40
    while True:
        if loop_calls > max_loop_calls:
            phi_alpha = func_in(x_in + alpha*p)
            while phi_alpha > phi_0 + mu_1*alpha*g.T@p:
                alpha = np.random.uniform(0, alpha)
                phi_alpha = func_in(x_in + alpha*p)
            alpha_star = alpha
            break
        phi_alpha = func_in(x_in + alpha*p)
        if (phi_alpha > phi_0 + mu_1*alpha*phi_prime_0) or (i > 1 and phi_alpha > phi_alpha_prev):
            alpha_star = pinpoint(func_in, x_in, g, p, mu_1, mu_2, alpha_prev, alpha)
            break
        phi_prime_alpha = find_gradient(func_in, (x_in + alpha*p)).T@p
        if abs(phi_prime_alpha) <= -mu_2*phi_prime_0:
            alpha_star = alpha
            break
        elif phi_prime_alpha >= 0:
            alpha_star = pinpoint(func_in, x_in, g, p, mu_1, mu_2, alpha, alpha_prev)
            break
        else:
            alpha_next = np.random.uniform(alpha, alpha_max)
        alpha_prev = alpha
        alpha = alpha_next
        phi_alpha_prev = phi_alpha
        loop_calls += 1
    return alpha_star

def pinpoint(func, x, g, p, mu_1, mu_2, alpha_low, alpha_high):
    phi_0 = func(x)
    phi_prime_0 = g.T@p
    loop_calls = 0
    max_loop_calls = 15
    while True:
        if loop_calls > max_loop_calls:
            # return np.random.uniform(np.minimum(alpha_low, alpha_high), np.maximum(alpha_low, alpha_high))
            return alpha_low*np.random.uniform()
        phi_alpha_low = func(x + alpha_low*p)
        # print("phi_alpha_low",phi_alpha_low, "for alpha_low:", alpha_low)
        phi_alpha_high = func(x + alpha_high*p)
        # print("phi_alpha_high",phi_alpha_high, "for alpha_high:", alpha_high)
        phi_prime_alpha_low = find_gradient(func, (x + alpha_low*p)).T@p
        alpha = ((2*alpha_low*(phi_alpha_high-phi_alpha_low)+phi_prime_alpha_low*(alpha_low**2-alpha_high**2))/
                (2*(phi_alpha_high-phi_alpha_low+phi_prime_alpha_low*(alpha_low-alpha_high))))
        phi_alpha = func(x + alpha*p)
        if (phi_alpha > phi_0 + mu_1*alpha*phi_prime_0) or (phi_alpha > phi_alpha_low):
            alpha_high = alpha
        else:
            phi_prime_alpha = find_gradient(func, (x + alpha*p)).T@p
            if abs(phi_prime_alpha) <= -mu_2*phi_prime_0:
                alpha_star = alpha
                break
            elif phi_prime_alpha*(alpha_high-alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha
        loop_calls += 1
    return alpha_star

def plot_brachistocrone(y):
    x = np.linspace(1,0,number_of_points)
    y = np.append(1,y)
    y = np.append(y,0)
    plt.plot(x,y)
    plt.show()

def plot_convergence():
    global convergences
    # print(convergences)
    norm_gradient = convergences[1:,0]
    function_call_count = convergences[1:,1]
    fig, ax = plt.subplots()
    ax.plot(function_call_count, norm_gradient)
    ax.set(xlabel='Function Calls', ylabel='Norm of the Gradient',
    #    title='Convergence Plot for the Matyas Function')
    #    title='Convergence Plot for the Rosenbrock Function')
       title='Convergence Plot for the Brachistocrone Function')
    plt.autoscale()
    # fig.savefig('matyas.eps', format='eps')
    # fig.savefig('rosenbrock.eps', format='eps')
    fig.savefig('brachistocrone.eps', format='eps')
    plt.show()

def optim_unc(func, x0):
    global out
    global convergences
    converged = 1e-6
    alpha_guess = 0.001 #TODO Change these as makes sense
    alpha_max = 0.1 #TODO Change these as makes sense
    x = x0
    delta = 10.0
    iterations =  0
    f_prev = func(x0)
    while delta > converged:
        # print(func(x))
        p = find_search_direction_steepest_decent(func, x)
        g = find_gradient(func, x)
        alpha = line_search_better(func, x, g, p, alpha_guess, alpha_max)
        x = x + alpha*p
        iterations += 1
        delta = abs(f_prev - func(x))
        f_prev = func(x)
        convergences = np.append(convergences,np.array([[np.linalg.norm(g),function_evals]]), axis = 0)
        print("Iterations", iterations, end = '\r') # ", Search Direction", p, ", x", x,
    print("\nFinal x:",x,"f(x):",func(x), "delta:", delta, "Function Evalutation:", function_evals)
    out = x

def example(x):
    return np.exp(x)/np.sqrt(np.sin(x)**3 + np.cos(x)**3)

def main():
    # x0 = np.array([1.000001,1.00001])
    # x0 = np.array([1.5,1.5])
    x0 = np.array([6.,-14.])
    # n = number_of_points
    # x0 = np.linspace(1,0,(n-2))
    func = matyas
    # func = rosenbrock
    # func = brachistochrone
    optim_unc(func, x0)
    # plot_convergence()
    # plot_brachistocrone(out)

if __name__ == '__main__':
    main()