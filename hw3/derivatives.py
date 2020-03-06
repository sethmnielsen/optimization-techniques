import numpy as np
from numpy import ndarray

import truss

def get_derivatives(method, A:ndarray):
    if method == 'FD':
        m, s, dm, ds = finite_diff_normal(A)
    elif method == 'complex':
        m, s, dm, ds = complex_step(A)
    elif method == 'AD':
        m, s, dm, ds = algo_diff(A)
    elif method == 'adjoint':
        m, s, dm, ds = adjoint(A)

    return m, s, dm, ds

def finite_diff(A_dict:dict, data:dict, h=1e-8):
    # m, s = truss(A)
    A = A_dict['areas']
    m = data['mass']
    s = data['stress_arr']
    n = len(A)
    dm = np.zeros(n)
    ds = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        mj, sj = truss.truss(A+e)
        # forward differencing
        dm[j] = (mj - m)/h
        ds[:,j] = (sj - s)/h
    
    # rng: np.random.Generator = np.random.default_rng()
    # dm = rng.random(10)
    funcs = {'mass': {'areas':dm},
             'stress_arr': {'areas':ds},
             'm': m,
             's': s}

    # return m, s, dm, ds
    return funcs, False

def finite_diff_normal(A, h=1e-8):
    m, s = truss.truss(A)
    
    n = len(A)
    dm = np.zeros(n)
    ds = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        mj, sj = truss.truss(A+e)
        # forward differencing
        dm[j] = (mj - m)/h
        ds[:,j] = (sj - s)/h
    
    # return m, s, dm, ds
    return m, s, dm, ds


def complex_step(A:ndarray, h=1e-40):
    m, s = truss.truss(A)

    n = len(A)
    dm = np.zeros(n)
    ds = np.zeros((n, n))
    hi = h*1.j
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = hi
        mj, sj = truss.truss(A+e)
        # complex step
        dm[j] = np.imag(mj)/h
        ds[:,j] = np.imag(sj)/h
    return m, s, dm, ds


def algo_diff(A:ndarray):
    import jax

    val_grad_mass = jax.value_and_grad(truss.mass_jax)
    jac_stresses = jax.jacfwd(truss.stress_jax)

    m, dm = val_grad_mass(A)
    s, ds = jac_stresses(A)

    return m, s, dm, ds

def adjoint(A_dict:dict, data:dict):
    # m, s = truss(A)
    A = A_dict['areas']
    m = data['mass']
    s = data['stress_arr']

    m, s, dm, ds = truss.adjoint(A)

    funcs = {'mass': {'areas':dm},
             'stress_arr': {'areas':ds},
             'm': m,
             's': s}

    # return m, s, dm, ds
    return funcs, False

def relative_errors(d_hat, d):
    ''' SHAPES
        d_hat: (3,p,n,c)
        d:     (1,p,n,c)

        OUT SHAPES - averages
        errors_mass:   (3,p,n)
        errors_stress: (3,p,n,n)
    '''
    _, p, n, c = d_hat.shape
    
    diff = d_hat-d
    E_mean = np.linalg.norm( np.mean(diff,axis=1), axis=(1,2) )
    E_max  = np.linalg.norm( np.max (diff,axis=1), axis=(1,2) )

    # d_norm_mean = np.linalg.norm( np.mean(  d, axis=1), axis=(1,2) )
    # d_norm_max  = np.linalg.norm( np.max (  d, axis=1), axis=(1,2) )
    # E_mean_rel = E_mean/d_norm_mean
    # E_max_rel  = (ds_arr - ds_true)/ds_true
    
    return E_mean, E_max 

    
def deriv_methods_compare(methods, A_list):
    p, n = A_list.shape
    dm_arr = np.zeros((4,p,n,1))
    ds_arr = np.zeros((4,p,n,n))
    m_arr = np.zeros((4,p))
    s_arr = np.zeros((4,p,n))
    for i, method in enumerate(methods):
        for j, A in enumerate(A_list):
            m, s, dm, ds = get_derivatives(method, A)
            dm_arr[i,j] = dm.reshape(n,1)
            ds_arr[i,j] = ds
            m_arr[i,j] = m
            s_arr[i,j] = s


    em_mean, em_max = relative_errors(dm_arr[:-1], dm_arr[-1])
    es_mean, es_max = relative_errors(ds_arr[:-1], ds_arr[-1])
    
    E_mean = np.array([em_mean, es_mean])
    E_max = np.array([em_max, es_max])
    
    return E_mean, E_max, [m_arr, s_arr, dm_arr, ds_arr]


if __name__ == '__main__':
    methods = ['FD', 'complex', 'AD', 'adjoint']
    n = 10  # number of truss elements
    p = 100 # number of trials
    rng = np.random.default_rng()
    A_ones = np.ones((1,n))
    A_rand = rng.uniform(0.1, 2.0, size=(p,n))
    
    E_mean_ones, E_max_ones, _ = deriv_methods_compare(methods, A_ones)
    E_mean_rand, E_max_rand, values = deriv_methods_compare(methods, A_rand)
    m_arr, s_arr, dm_arr, ds_arr = values

    np.set_printoptions(precision=8, linewidth=200, floatmode='fixed', suppress=False)
    print(f'\nmass avg: {np.mean(m_arr)}, max: {np.max(m_arr)}')
    print(f'stress avg: {np.mean(s_arr)}, max: {np.max(s_arr)}')
    print(f'\ndm avg: {np.mean(dm_arr)}, max: {np.max(dm_arr)}')
    print(f'ds avg: {np.mean(ds_arr)}, max: {np.max(ds_arr)}')

    print(f'\n-- A = np.ones(10)')
    print(f'mean mass error for each method: {E_mean_ones[0]}')
    print(f'mean stress error for each method: {E_mean_ones[1]}')

    print(f'\n-----MEAN VALUES-----')

    print(f'\n-- Random areas')
    print(f'mean mass error for each method: {E_mean_rand[0]}')
    print(f'mean stress error for each method: {E_mean_rand[1]}')

    print(f'\n\n-----MAX VALUES-----')

    print(f'\n-- Random areas')
    print(f'max mass error for each method: {E_max_rand[0]}')
    print(f'max stress error for each method: {E_max_rand[1]}\n')