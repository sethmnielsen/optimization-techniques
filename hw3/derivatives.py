import numpy as np
from numpy import ndarray

from truss import *

def get_derivatives(method, A:ndarray):
    if method == 'FD':
        m, s, dm, ds = finite_diff(A)
    elif method == 'complex':
        m, s, dm, ds = complex_step(A)
    elif method == 'AD':
        m, s, dm, ds = algo_diff(A)
    elif method == 'adjoint':
        m, s, dm, ds = adjoint(A)

    return m, s, dm, ds

def finite_diff(A:ndarray, h=1e-8):
    m, s = truss(A)

    n = len(A)
    dm = np.zeros(n)
    ds = np.zeros((n,s.size))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        mj, sj = truss(A+e)
        # forward differencing
        dm[j] = (mj - m)/h
        ds[j] = (sj - s)/h
    return m, s, dm, ds

def complex_step(A:ndarray, h=1e-40):
    m, s = truss(A)

    n = len(A)
    dm = np.zeros(n)
    ds = np.zeros((n, s.size))
    hi = h*1.j
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = hi
        mj, sj = truss(A+e)
        # complex step
        dm[j] = np.imag(mj)/h
        ds[j] = np.imag(sj)/h
    return m, s, dm, ds


def algo_diff(A:ndarray):
    import jax

    val_grad_mass = jax.value_and_grad(truss_mass_jax)
    jac_stresses = jax.jacfwd(truss_stress_jax)

    m, dm = val_grad_mass(A)
    s, ds = jac_stresses(A)

    return m, s, dm, ds

def adjoint(A:ndarray):
    m, s, dm, ds = truss_adjoint(A)

    return m, s, dm, ds

def relative_errors(dm_arr, ds_arr, dm_true, ds_true):
    ''' SHAPES
        dm_arr:  (3,p,n)
        ds_arr:  (3,p,n,n)
        dm_true: (1,p,n)
        ds_true: (1,p,n,n)

        OUT SHAPES - averages
        errors_mass:   (3,p,n)
        errors_stress: (3,p,n,n)
    '''
    errors_mass = (dm_arr - dm_true)/dm_true
    errors_stress = (ds_arr - ds_true)/ds_true

    return errors_mass, errors_stress


if __name__ == '__main__':
    methods = ['FD', 'complex', 'AD', 'adjoint']
    n = 10  # number of truss elements
    p = 100 # number of trials
    # rng = np.random.default_rng()
    # A_list = rng.uniform(0.5, 1.5, size=(p,n))
    p = 1
    A_list = np.ones((p,n))

    dm_arr = np.zeros((4,p,n))
    ds_arr = np.zeros((4,p,n,n))
    m_arr = np.zeros((4,p))
    s_arr = np.zeros((4,p,n))
    for i, method in enumerate(methods):
        for j, A in enumerate(A_list):
            m, s, dm, ds = get_derivatives(method, A)
            dm_arr[i,j] = dm
            ds_arr[i,j] = ds
            m_arr[i,j] = m
            s_arr[i,j] = s

    em, es = relative_errors(dm_arr[:-1], ds_arr[:-1], dm_arr[-1], ds_arr[-1])

    # np.set_printoptions(linewidth=200, precision=20)
    np.set_printoptions(precision=6, linewidth=200, floatmode='fixed', suppress=False)
    print(f'\nmass avg: {np.mean(m_arr)}, max: {np.max(m_arr)}')
    print(f'stress avg: {np.mean(s_arr)}, max: {np.max(s_arr)}')
    print(f'\ndm avg: {np.mean(dm_arr)}, max: {np.max(dm_arr)}')
    print(f'ds avg: {np.mean(ds_arr)}, max: {np.max(ds_arr)}')
    print(f'\n-----MEAN VALUES-----')
    # print(f'\nmean error mass:\n{em_mean}')
    # print(f'\nmean error stress:\n{es_mean}')
    print(f'\nmean mass error for each method: {np.mean(em,axis=(1,2))}')
    print(f'mean stress error for each method: {np.mean(es,axis=(1,2,3))}')

    print(f'\n\n-----MAX VALUES-----')
    # print(f'\nmax error mass:\n{em_max}')
    # print(f'\nmax error stress:\n{es_max}')
    print(f'\nmax mass error for each method: {np.max(em,axis=(1,2))}')
    print(f'max stress error for each method: {np.max(es,axis=(1,2,3))}\n')

    # print(f'\n\n\ndm: \n{dm_avg}')
    # print(f'\n\nds: \n{ds_avg}')