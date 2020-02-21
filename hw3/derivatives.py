import numpy as np
from numpy import ndarray

from truss import *

def get_derivatives(method, A:ndarray, m:float, s:ndarray):
    if method == 'FD':
        dm, ds = finite_diff(A, m, s)
    elif method == 'complex':
        dm, ds = complex_step(A, m, s)
    elif method == 'AD':
        # jnp.array(A)
        dm, ds = algo_diff(A, m, s)
    elif method == 'adjoint':
        dm, ds = adjoint(A, m, s)

    return dm, ds

def finite_diff(A:ndarray, m:float, s:ndarray, h=1e-8) -> (ndarray):
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
    return dm, ds

def complex_step(A:ndarray, m:float, s:ndarray, h=1e-40):
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
    return dm, ds


def algo_diff(A:ndarray, m:float, s:ndarray):
    import jax

    grad_mass = jax.grad(truss_mass_jax)
    grad_stresses = jax.jacfwd(truss_stress_jax)

    dm = grad_mass(A)
    ds = grad_stresses(A)

    return dm, ds

def adjoint(A:ndarray, m:float, s):
    return 0, 0


if __name__ == '__main__':
    # import jax.numpy as np
    A = np.ones(10)
    m, s = truss(A)
    method = 'AD'

    dm, ds = get_derivatives(method, A, m, s)

    print(f'\ndm: {dm}')
    print(f'\nds: {ds}')