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
        dm_arr:  (3,10)
        ds_arr:  (3,10,10)
        dm_true: (1,10)
        ds_true: (1,10,10)

        OUT SHAPES - averages
        errors_mass:   (10)
        errors_stress: (10,10)
    '''
    errors_mass = np.mean( (dm_arr - dm_true)/dm_true, axis=1 )
    errors_stress = np.mean( (ds_arr - ds_true)/ds_true, axis=1 )

    return errors_mass, errors_stress


if __name__ == '__main__':
    A_list = np.zeros((10,10))
    A_list[0] = np.ones(10)
    A_list[1:] = np.random.randn(9,10)
    methods = ['FD', 'complex', 'AD', 'adjoint']

    dm_arr = np.zeros((4,10,10))
    ds_arr = np.zeros((4,10,10,10))
    for i, method in enumerate(methods):
        for j, A in enumerate(A_list):
            m, s, dm, ds = get_derivatives(method, A)
            dm_arr[i,j] = dm
            ds_arr[i,j] = ds

    em, es = relative_errors(dm_arr[:-1], ds_arr[:-1], dm_arr[-1], ds_arr[-1])

    dm_avg = np.mean(dm_arr, axis=1)
    ds_avg = np.mean(ds_arr, axis=1)

    np.set_printoptions(linewidth=200)
    print(f'\n-----MEAN VALUES-----')
    print(f'\n\nerrors_mass:\n{em}')
    print(f'\nerrors_stress:\n{es}')

    # print(f'\nmasses: {m_np_avg}')
    # print(f'\nstress arrays: {s_np_avg}')
    print(f'\n\n\ndm: \n{dm_avg}')
    print(f'\n\nds: \n{ds_avg}')