import numpy as np 
import pyoptsparse
import matplotlib.pyplot as plt

#Consider reformatting with my derivatives
l = 1.5
t0 = 0.0
tf = 15.0
use_gamma = True

def position(t, p):
    px = p[:6]
    py = p[6:]
    tvec = np.stack([t**0,
                     t**1,
                     t**2,
                     t**3,
                     t**4,
                     t**5])
    
    x = px @ tvec 
    y = py @ tvec 

    return x, y

def velocity(t, p):
    px = p[:6]
    py = p[6:]
    tvec = np.stack([0 * t**0,
                     1 * t**0,
                     2 * t**1,
                     3 * t**2,
                     4 * t**3,
                     5 * t**4])
    
    vx = px @ tvec 
    vy = py @ tvec 
    return vx, vy

def acceleration(t, p):
    px = p[:6]
    py = p[6:]
    tvec = np.stack([0 * t**0,
                     0 * t**0,
                     2 * t**0,
                     6 * t**1,
                     12 * t**2,
                     20 * t**3])
    
    ax = px @ tvec 
    ay = py @ tvec 
    return ax, ay

def jerk(t, p):
    px = p[:6]
    py = p[6:]
    tvec = np.stack([0 * t**0,
                     0 * t**0,
                     0 * t**0,
                     6 * t**0,
                     24 * t**1,
                     60 * t**2])
    
    Jx = px @ tvec 
    Jy = py @ tvec 

    return Jx, Jy

def equalityConstraints(t0, tf, p):
    x0, y0 = position(t0, p)
    xf, yf = position(tf, p)
    vx0, vy0 = velocity(t0, p)
    vxf, vyf = velocity(tf, p)

    eq_con = np.zeros(8, dtype=type(p[0]))
    eq_con[0] = x0 
    eq_con[1] = y0 
    eq_con[2] = vx0 
    eq_con[3] = vy0 
    eq_con[4] = xf 
    eq_con[5] = yf 
    eq_con[6] = vxf 
    eq_con[7] = vyf 

    return eq_con

def inequalityConstraints(t, p):
    vx, vy = velocity(t, p)
    ax, ay = acceleration(t, p)

    v = np.sqrt(vx**2 + vy**2)
    a = np.sqrt(ax**2 + ay**2)

    arg1 = v/l * np.tan(np.pi/4)
    arg2 = (vx * ay - vy * ax) / v**3
    gmax = arg1 - arg2
    gmax2 = arg1 + arg2
    
    if use_gamma:
        ineq_con = np.zeros(400, dtype=type(p[0]))
        ineq_con[:100] = 10**2 - v**2
        ineq_con[100:200] = 2**2 - a**2
        ineq_con[200:300] = gmax 
        ineq_con[300:] = gmax2
    else:
        ineq_con = np.zeros(200, dtype=type(p[0]))
        ineq_con[:100] = 10**2 - v**2
        ineq_con[100:] = 2**2 - a**2

    return ineq_con

def objective(x_dict):
    p = x_dict['xvars']
    t_span = np.linspace(t0, tf, 100)
    funcs = {}

    #Implement integration of cost function here
    Jx, Jy = jerk(t_span, p)
    vdd = Jx**2 + Jy**2
    funcs['obj'] = np.sum(vdd) # I'm not sure that squaring vdd will do anything...

    #Equality Constraints
    funcs['eq_con'] = equalityConstraints(t0, tf, p)

    #Inequality Constraints
    funcs['ineq_con'] = inequalityConstraints(t_span, p)
    
    return funcs, False

if __name__=="__main__": 
    p = np.ones(12) * 1e-4

    optProb = pyoptsparse.Optimization('Differential_Flatness', objective)
    optProb.addVarGroup('xvars', 12, 'c', lower=None, upper=None, value=p)
    eq_bnd = [0.0, 0.0, 0.0, 2.0, 10.0, 0.0, 0.0, 1.0]
    optProb.addConGroup('eq_con', 8, lower=eq_bnd, upper=eq_bnd) 
    if use_gamma:
        optProb.addConGroup('ineq_con', 400, lower=0.0, upper=None)  # I can get an answer but the constraints are violated and it runs into numerical difficulties
    else:
        optProb.addConGroup('ineq_con', 200, lower=0.0, upper=None)  
    optProb.addObj('obj')

    opt = pyoptsparse.SNOPT()
    sol = opt(optProb, sens='CS', storeHistory='constrained.txt') 
    print(sol.xStar['xvars'])
    print('Optimum Value: ', sol.fStar.item(0))

    # hist = pyoptsparse.History('constrained.txt', flag='r') #Has the history info
    data = np.loadtxt('SNOPT.dat', skiprows=3) #had to edit SNOPT.dat to work
    opt = data[:,2]
    feas = data[:,3]
    idx = np.arange(0, opt.size)

    p_star = sol.xStar['xvars']

    t_vec = np.linspace(t0, tf, 100)
    x, y = position(t_vec, p_star)
    vx, vy = velocity(t_vec, p_star)
    ax, ay = acceleration(t_vec, p_star)
    v = np.sqrt(vx**2 + vy**2)
    theta = np.rad2deg(np.arctan2(vy, vx))
    gamma = np.rad2deg(np.arctan2(l * (vx * ay - vy * ax), v**3)) 

    print('Checking Constraints')
    print('x0: {}, y0: {}, xf: {}, yf: {}'.format(x[0], y[0], x[-1], y[-1]))
    print('vx0: {}, vy0: {}, vxf: {}, vyf: {}'.format(vx[0], vy[0], vx[-1], vy[-1]))
    print('Max v: {}'.format(np.max(v)))
    print('Max a: {}'.format(np.max(ax**2 + ay**2)))
    print('Max gamma: {}, Min gamma: {}'.format(np.max(gamma), np.min(gamma)))

    plt.figure(1)
    plt.plot(x, y)
    plt.title('Trajectory')
    plt.axis('equal')
    plt.savefig('trajectory.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

    fig2, ax = plt.subplots(3,1, sharex=True)
    ax[0].set_title('States')
    ax[0].plot(t_vec, x)
    ax[0].set_ylabel('x (m)')
    ax[1].plot(t_vec, y)
    ax[1].set_ylabel('y (m)')
    ax[2].plot(t_vec, theta)
    ax[2].set_ylabel(r'$\theta$ (rad)')
    ax[2].set_xlabel('Time (s)')
    plt.savefig('states.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

    fig3, ax3 = plt.subplots(2,1, sharex=True)
    ax3[0].set_title('Inputs')
    ax3[0].plot(t_vec, v)
    ax3[0].set_ylabel('V (m/s)')
    ax3[1].plot(t_vec, gamma)
    ax3[1].set_ylabel('$\gamma$ (rad)')
    ax3[1].set_xlabel('Time (s)')
    plt.savefig('inputs.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

    fig4, ax4 = plt.subplots(2,1, sharex=True)
    ax4[0].plot(idx, opt)
    ax4[1].set_xlabel('Major Iterations')
    ax4[0].set_ylabel('Optimality')
    ax4[1].plot(idx, feas)
    ax4[1].set_ylabel('Feasibility')
    plt.savefig('optimality.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()