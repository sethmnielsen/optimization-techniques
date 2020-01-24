import pyoptsparse as pyop


def objfunc(xdict):
    x = xdict['xvars']
    funcs = {}
    funcs['obj'] = -x[0]*x[1]*x[2]
    conval = [0]*2
    conval[0] = x[0] + 2.*x[1] + 2.*x[2] - 72.0
    conval[1] = -x[0] - 2.*x[1] - 2.*x[2]
    funcs['con'] = conval
    fail = False

    return funcs, fail

optProb: pyop.Optimization = pyop.Optimization('TP037', objfunc)

# Design variables - can be created individually or as a group
optProb.addVarGroup('xvars', nVars=3, type='c', lower=[0,0,0], upper=[42,42,42], value=10)

# Constraints - also individually or group
optProb.addConGroup('con', nCon=2, lower=None, upper=0.0)

optProb.addObj('obj')
print(optProb)

opt = pyop.SNOPT()
opt.setOption('iPrint', 0)
opt.setOption('iSumm', 0)

sol = opt(optProb, sens='FD')
print(sol)