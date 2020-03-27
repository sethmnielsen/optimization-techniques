# %%
from pyoptsparse import History

h = History('/home/seth/school/optimization/output/hw5.hst', flag='r')

# %%

# Convert db to dict
d = {}
for key in h.keys:
    d[key] = h.readData(key)

# %%

varinfo = d['varInfo']
conInfo = d['conInfo']
xs = d['xs']
hs = d['hs']

data = [] 
for i in range(int(d['last'])+1):
    data.append( d[str(i)] )
    
# %%
for i in range(8):
    print(f'{i}: {data[i].values()}')

# %%
xuser = np.zeros((8,4,6))
xuseri = np.zeros((4,6))
funcs = np.zeros((8,4,4))
funcsi = np.zeros((4,4))
for i in range(len(data)):
    xuser_dict = data[i]['xuser']
    for j, p in enumerate(xuser_dict.values()):
        xuseri[j] = p
    keys = data[i].keys()
    fkey = 'funcs' if 'funcs' in keys else 'funcsSens'
    funcs_dict = data[i][fkey]
    print(f'{funcs_dict = }')
    for j, con in enumerate(funcs_dict.values()):
        print(j)
        if j == 4:
            break
        funcsi[j] = con
    funcs[i] = funcsi

# %%
