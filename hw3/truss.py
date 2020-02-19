import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) # make sure jax always uses doubles

import numpy as np
from numpy import ndarray
from IPython.core.debugger import Pdb

'''
Computes mass and stress for the 10-bar truss structure
Parameters:
A: array of length 10 w/ the cross sectional area of each bar
    (see image in hw writeup for number order if interested)

Outputs:
mass: float mass of the entire structure
stress: array of length 10 with corresponding stress in each bar
'''
def truss(A: ndarray):
    typ = A.dtype
    P = 1e5 # applied loads
    Ls = 360 # length of sides
    Ld = np.sqrt(Ls**2 * 2) # length of diagonals

    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4], dtype=typ) - 1
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1], typ) - 1
    phi = np.radians(np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45])).astype(typ)
    L1 = np.array([Ls]*6, typ)
    L2 = np.array([Ld]*4, typ)
    L = np.concatenate((L1, L2), axis=0)

    nbar = np.size(A)
    E = 1e7 * np.ones(nbar, typ) # modulus of elasticity
    rho = 0.1 * np.ones(nbar, typ) # material density

    Fx = np.zeros(6, dtype=typ)
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0], typ)
    rigidx = np.array([0, 0, 0, 0, 1, 1], typ)
    rigidy = np.array([0, 0, 0, 0, 1, 1], typ)

    n = np.size(Fx) #number of nodes
    DOF = 2

    # compute mass
    mass = np.sum(rho * A * L)

    # assemble global matrices
    K: ndarray = np.zeros((DOF * n, DOF * n), typ)
    S: ndarray = np.zeros((nbar, DOF * n), typ)

    for i in range(nbar): #loop through each bar
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        idx = node2idx([start[i], finish[i]], DOF)
        idxx, idxy = np.meshgrid(idx,idx)

        K[idxy, idxx] += Ksub
        S[i, idx] = Ssub

    #setup applied loads
    F = np.zeros(n * DOF, typ)

    for i in range(n):
        idx = node2idx([i], DOF)
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    #setup boundary condition
    idxx = np.nonzero(rigidx)[0]
    idxy = np.nonzero(rigidy)[0]
    removex = node2idx(idxx.tolist(), DOF)
    tempx = np.reshape(removex, (2,-1), order='F')
    removey = node2idx(idxy.tolist() , DOF)
    tempy = np.reshape(removey, (2,-1), order='F')
    removex = tempx[0,:]
    removey = tempy[1,:]

    remove = np.concatenate((removex, removey), axis=0)

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove)
    S = np.delete(S, remove, axis=1)

    d = np.linalg.solve(K, F)
    stress = S @ d

    return mass, stress

def truss_mass(A):
    Ls = 360 # length of sides
    Ld = np.sqrt(Ls**2 * 2)  # length of diagonals

    L = np.concatenate(([Ls] * 6, [Ld] * 4), axis=0)

    nbar = np.size(A)
    rho = 0.1 * np.ones(nbar) # material density

    # compute mass
    mass = np.sum(rho * A * L)

    return mass

def truss_stress(A):
    P = 1e5 # applied loads
    Ls = 360 # length of sides
    Ld = np.sqrt(360**2 * 2) # length of diagonals

    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4]) - 1
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1]) - 1
    phi = np.deg2rad(np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45]))
    L = np.concatenate(([Ls] * 6, [Ld] * 4), axis=0)

    nbar = np.size(A)
    E = 1e7 * np.ones( nbar) # modulus of elasticity

    Fx = np.zeros(6, dtype=np.float64)
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigidx = np.array([0, 0, 0, 0, 1, 1])
    rigidy = np.array([0, 0, 0, 0, 1, 1])

    n = np.size(Fx) #number of nodes
    DOF = 2

    # assemble global matrices
    K = np.zeros((DOF * n, DOF * n))
    S = np.zeros((nbar, DOF * n))

    for i in range(nbar): #loop through each bar
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        idx = node2idx([start[i], finish[i]], DOF)
        idxx, idxy = np.meshgrid(idx,idx)

        K[idxy, idxx] += Ksub
        S[i, idx] = Ssub

    #setup applied loads
    F = np.zeros(n * DOF)

    for i in range(n):
        idx = node2idx([i], DOF)
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    #setup boundary condition
    idxx = np.nonzero(rigidx)[0]
    idxy = np.nonzero(rigidy)[0]
    removex = node2idx(idxx.tolist(), DOF)
    tempx = np.reshape(removex, (2,-1), order='F')
    removey = node2idx(idxy.tolist() , DOF)
    tempy = np.reshape(removey, (2,-1), order='F')
    removex = tempx[0,:]
    removey = tempy[1,:]

    remove = np.concatenate((removex, removey), axis=0)

    K = onp.delete(K, remove, axis=0)
    K = onp.delete(K, remove, axis=1)
    F = onp.delete(F, remove)
    S = onp.delete(S, remove, axis=1)

    d = np.linalg.solve(K, F)
    stress = S @ d

    return stress



'''
Compute the stiffness and stress matrix for one element
Parameters:
E: float: modulus of elasticity
A: float: cross sectional area
L: float: length of element
phi: float: orientation of element

Outputs:
K: 4 x 4 ndarray: stiffness matrix
S: 1x4 ndarray: stress matrix
'''
def bar(E, A, L, phi):
    c = np.cos(phi)
    s = np.sin(phi)

    k0 = np.array([[c**2, c*s], [c*s, s**2]])

    K = E * A / L * np.block([[k0, -1*k0], [-1*k0, k0]])

    S = E / L * np.array([-c, -s, c, s])

    return K, S

'''
Computes the appropriate indices in the global matrix for
the corresponding node numbers. You pass in the number of the node
(either as a scalar or an array of locations), and the DOF per node
and it returns the corresponding indices in the global matrices
'''
def node2idx(node, DOF):
    idx = np.empty(0)

    for i in range(len(node)):
        start = DOF * (node[i]-1) + 2
        finish = DOF * node[i] + 1

        idx = np.hstack((idx, np.arange(start, finish+1)))

    return idx.astype(int)

if __name__=="__main__":
    temp = np.ones(10) * .1
    mass, stress = truss(temp)
    print(mass, stress)