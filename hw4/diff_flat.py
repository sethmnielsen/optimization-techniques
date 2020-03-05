# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Differential Flatness of a Moving Car
# 

# %%
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=22)
rc('figure', figsize=(8,6))
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize


# %%
from IPython.display import display
from sympy import init_printing
init_printing(use_latex='mathjax')
import sympy as sp


from sympy.physics.vector import dynamicsymbols
t = sp.symbols('t')
display(sp.Integral(sp.sqrt(1/t), t))


x = dynamicsymbols('x_d')
xd = sp.diff(x, t)
y = dynamicsymbols('y_d')
yd = sp.diff(y, t)

v = sp.sqrt(xd**2 + yd**2)
vd = sp.diff(v, t)
vdd = sp.diff(vd, t)
display(vd)
display(vdd)


# %%
xd, xdd, xddd, yd, ydd, yddd = sp.symbols('xd, xdd, xddd, yd, ydd, yddd')
vdd = 2*xdd**2 + 2*xd*xddd + 2*ydd**2 + 2*yd*yddd
vddsq = sp.expand(vdd**2)
print(vddsq)


# %%
# car parameters
ell = 2.5

# time
ts = 0.
te = 20.

npts = 100 # number of points to discretize at

# time
tvec = np.arange(ts, te, 1./npts)

# parameters 0:6 are p_xi 6:12 are p_yi
nparam = 6
ndim = 2
P0 = np.zeros((nparam*ndim))


# path derivatives
def P_0d(P, t, p0):
    PP = P[p0:nparam+p0]
    TT = np.stack([ t**0,
                    t**1,
                    t**2,
                    t**3,
                    t**4,
                    t**5])
    func = PP@TT
    return func

def P_1d(P, t, p0):
    PP = P[p0:nparam+p0]
    TT = np.stack([ 0*t**0,
                    1*t**0,
                    2*t**1,
                    3*t**2,
                    4*t**3,
                    5*t**4])
    func = PP@TT
    return func

def P_2d(P, t, p0):
    PP = P[p0:nparam+p0]
    TT = np.stack([ 0*t**0,
                    0*t**0,
                    2*t**0,
                    6*t**1,
                    12*t**2,
                    20*t**3])
    func = PP@TT
    return func

def P_3d(P, t, p0):
    PP = P[p0:nparam+p0]
    TT = np.stack([ 0*t**0,
                    0*t**0,
                    0*t**0,
                    6*t**0,
                    24*t**1,
                    60*t**2])
    func = PP@TT
    return func


# objective function
def Jmin(P):
    
#    xd   = P_1d(P, tvec, 0)
#    xdd  = P_2d(P, tvec, 0)
    xddd = P_3d(P, tvec, 0)
    
#    yd   = P_1d(P, tvec, nparam)
#    ydd  = P_2d(P, tvec, nparam)
    yddd = P_3d(P, tvec, nparam)
    
    vdd = xddd**2 + yddd**2
    
    J = sum(vdd**2)
    return J
    

# %%
# constraints
def maxSteer(P):
    xd   = P_1d(P, tvec, 0)
    xdd  = P_2d(P, tvec, 0) 
    yd   = P_1d(P, tvec, nparam)
    ydd  = P_2d(P, tvec, nparam)
    
    v = np.sqrt(xd**2 + yd**2)
    gmax = np.min(v*np.tan(np.pi/4)/ell - np.abs(ydd*xd + yd*xdd))
    #import pdb; pdb.set_trace()
    return gmax
    
constraintList = [
# x(0) = (0, 0)
    {'type': 'eq', 'fun': lambda P: P_0d(P, np.array([ts]), 0)},
    {'type': 'eq', 'fun': lambda P: P_0d(P, np.array([ts]), nparam)},
# x(5) = (5, 3)
    #{'type': 'eq', 'fun': lambda P: P_0d(P, np.array([5]), 0) - 5},
    #{'type': 'eq', 'fun': lambda P: P_0d(P, np.array([5]), nparam) - 2},
# x(te) = (10, 5)
    {'type': 'eq', 'fun': lambda P: P_0d(P, np.array([te]), 0) - 10},
    {'type': 'eq', 'fun': lambda P: P_0d(P, np.array([te]), nparam) - 5},
# xd(0) = (0, 1)
    {'type': 'eq', 'fun': lambda P: P_1d(P, np.array([ts]), 0) - 0},
    {'type': 'eq', 'fun': lambda P: P_1d(P, np.array([ts]), nparam) - 2},
# xd(te) = (0, 0)
    {'type': 'eq', 'fun': lambda P: P_1d(P, np.array([te]), 0) },
    {'type': 'eq', 'fun': lambda P: P_1d(P, np.array([te]), nparam) - 1},
# Max vel < 10 m/s
    {'type': 'ineq', 'fun': lambda P: 10**2 - np.amax(P_1d(P, tvec, 0)**2 + P_1d(P, tvec, nparam)**2)},
# Max accel < 2 m/ss
    {'type': 'ineq', 'fun': lambda P: 2**2 - np.amax(P_2d(P, tvec, 0)**2 + P_2d(P, tvec, nparam)**2)},
# Max steer < 45 deg
    {'type': 'ineq', 'fun': maxSteer}
]


# %%
def cb(state):
    pass
    #print(state)
    
res = minimize(Jmin, P0, method='SLSQP', constraints=constraintList, callback=cb)
print(res)


# %%
# use the result!
Popt = res.x

xtraj = P_0d(Popt, tvec, 0)
ytraj = P_0d(Popt, tvec, nparam)

xtraj_dot = P_1d(Popt, tvec, 0)
ytraj_dot = P_1d(Popt, tvec, nparam)

xtraj_ddot = P_2d(Popt, tvec, 0)
ytraj_ddot = P_2d(Popt, tvec, nparam)

thetatraj = np.arctan2(ytraj_dot, xtraj_dot)

vtraj = np.sqrt(xtraj_dot**2 + ytraj_dot**2)
gamtraj = np.arctan2(ell * (ytraj_ddot*xtraj_dot - ytraj_dot*xtraj_ddot), vtraj)

fig, ax = plt.subplots(3, 1)
ax[0].set_title(r"$\mathbf{x}_d$ vs. $t$")
ax[0].plot(tvec,xtraj, label=r"$x_d$", linewidth=2)
ax[0].legend()
ax[1].plot(tvec,ytraj, label =r"$y_d$", linewidth=2)
ax[1].legend()
ax[2].plot(tvec,np.rad2deg(thetatraj), label =r"$\theta_d$", linewidth=2)
ax[2].legend()
fig.tight_layout(pad=.2)
#fig.savefig('car_df_states.png', dpi=600)

fig, ax = plt.subplots(2, 1)
ax[0].set_title(r"$\mathbf{u}_d$ vs $t$")
ax[0].plot(tvec, vtraj, label=r"$v_d$", linewidth=2)
ax[0].legend()
ax[1].plot(tvec, np.rad2deg(gamtraj), label=r"$\gamma_d$", linewidth=2)
ax[1].legend()
#fig.savefig('car_df_inputs.png', dpi=600)

fig, ax = plt.subplots(1, 1)
ax.set_title(r"$x_d$ vs $y_d$")
ax.plot(xtraj, ytraj, linewidth=2, color='r')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.axis('equal')
#fig.savefig('car_df_xy.png', dpi=600)


plt.show()


# %%


