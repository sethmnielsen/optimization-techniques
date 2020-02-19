def truss(A):
    P = 1e5 # applied loads
    Ls = 360 # length of sides
    Ld = np.sqrt(360**2 * 2) # length of diagonals

    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4]) - 1
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1]) - 1
    phi = np.deg2rad(np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45]))
    L1 = np.array([Ls]*6)
    L2 = np.array([Ld]*4)
    L = np.concatenate((L1, L2), axis=0)

    nbar = np.size(A)
    E = 1e7 * np.ones(nbar) # modulus of elasticity
    rho = 0.1 * np.ones(nbar) # material density

    Fx = np.zeros(6, dtype=np.float64)
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigidx = np.array([0, 0, 0, 0, 1, 1])
    rigidy = np.array([0, 0, 0, 0, 1, 1])

    n = np.size(Fx) #number of nodes
    DOF = 2

    # compute mass
    mass = np.sum(rho * A * L)

    # assemble global matrices
    K: np.DeviceArray = np.zeros((DOF * n, DOF * n))
    S: np.DeviceArray = np.zeros((nbar, DOF * n))

    Knp: ndarray = onp.zeros((DOF * n, DOF * n))
    Snp: ndarray = onp.zeros((nbar, DOF * n))
    for i in range(nbar): #loop through each bar
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        idx = node2idx([start[i], finish[i]], DOF)
        idxx, idxy = onp.meshgrid(idx,idx)

        Knp[idxy, idxx] += Ksub

        Snp[i, idx] = Ssub

    #setup applied loads
    Fnp: ndarray = onp.zeros(n * DOF)

    for i in range(n):
        idx = node2idx([i], DOF)
        Fnp[idx[0]] = Fx[i]
        Fnp[idx[1]] = Fy[i]


    #setup boundary condition
    idxx = np.nonzero(rigidx)[0]
    idxy = np.nonzero(rigidy)[0]
    removex = node2idx(idxx.tolist(), DOF)
    tempx = np.reshape(removex, (2,-1), order='F')
    removey = node2idx(idxy.tolist() , DOF)
    tempy = np.reshape(removey, (2,-1), order='F')
    removex = tempx[0,:]
    removey = tempy[1,:]

    remove = onp.concatenate((removex, removey), axis=0)

    Knp = onp.delete(Knp, remove, axis=0)
    Knp = onp.delete(Knp, remove, axis=1)
    Fnp = onp.delete(Fnp, remove)
    Snp = onp.delete(Snp, remove, axis=1)

    K = np.array(Knp)
    S = np.array(Snp)
    F = np.array(Fnp)

    d = np.linalg.solve(K, F)
    stress = S @ d

    return mass, stress