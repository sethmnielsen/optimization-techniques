import numpy as np

def f(x):
    d = np.exp(x) / np.sqrt( np.sin(x)**3 + np.cos(x)**3 )
    return d
    
x = 2.0
h = 0.00000000001

df_dx = (f(x+h) - f(x)) / h
print(df_dx)


def f_prime(x):
    nom = np.exp(x)*(3*np.sin(x)**2*np.cos(x) - 3*np.sin(x)*np.cos(x)**2)
    denom = 2*(np.sin(x)**3 + np.cos(x)**3)**(3/2)
    return nom/denom
    
fp = f(x) - f_prime(x)
print(fp)