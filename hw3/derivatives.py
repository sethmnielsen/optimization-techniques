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
    dm = np.zeros(n, dtype=np.complex128)
    ds = np.zeros((n, s.size), dtype=np.complex128)
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
    errors_mass = np.zeros(3)
    errors_mass = np.mean( (dm_arr - dm_true)/dm_true, axis=1 )
    errors_stress = np.mean( (ds_arr - ds_true)/ds_true, axis=1 )

    return errors_mass, errors_stress


if __name__ == '__main__':
    A_list = [np.ones(10)]
    A_list.append(np.random.randn(9))
    methods = ['FD', 'complex', 'AD', 'adjoint']

    errors_mass = []
    errors_stress = []
    dm_arr = np.zeros((4,10))
    ds_arr = np.zeros((4,10,10))
    # filename = 'derivative errors'
    # with expression as target:
        # pass
    for i, method in enumerate(methods):
        mass_arr = []
        stress_arr = []
        for j, A in enumerate(A_list):
            m, s, dm, ds = get_derivatives(method, A)
            # mass_arr.append(m)
            # stress_arr.append(s)
            dm_arr[i,j] = dm
            ds_arr[i,j] = ds
        # m_np.append(np.array(mass_arr))
        # s_np.append(np.array(stress_arr))
        # errors_mass.append(em)
        # errors_stress.append(es)

    em, es = relative_errors(dm_arr[:-1], ds_arr[:-1], dm_arr[-1], ds_arr[-1])
    dm_avg = np.mean(dm_arr, axis=1)
    ds_avg = np.mean(ds_arr, axis=1)
    # err_m_avg = np.mean(errors_mass)
    # err_s_avg = np.mean(errors_stress)

    print(f'\n-----MEAN VALUES-----')
    print(f'errors_mass: {em}')
    print(f'\nerrors_stress: {es}')

    # print(f'\nmasses: {m_np_avg}')
    # print(f'\nstress arrays: {s_np_avg}')
    print(f'\ndm: {dm_avg}')
    print(f'\nds: {ds_avg}')