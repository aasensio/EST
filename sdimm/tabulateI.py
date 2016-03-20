import numpy as np
import scipy.integrate
import matplotlib.pyplot as pl
from ipdb import set_trace as stop

def func(theta, u, S, phi):    
    t1 = u*(1.0/8.0 * np.arccos(u) + np.sqrt(1.0-u**2) * ((u**3 / 12.0 - 5.0*u/24.0) + (u**3/3.0 - u/3.0) * np.cos(theta)**2))
    t2 = (S**2 + 2*S*u*np.cos(theta+phi) + u**2)**(5.0/6.0)
    t3 = (S**2 - 2*S*u*np.cos(theta+phi) + u**2)**(5.0/6.0) - 2.0*u**(5.0/3.0)

    return (16.0 / np.pi)**2 * t1 * (t2+t3)

def funI(S, phi):
    return scipy.integrate.dblquad(func, 0.0, 1.0, lambda x: 0.0, lambda x: 2.0*np.pi, args=(S,phi))

x = np.linspace(0.0, 50.0, 1001)
value = np.zeros((2,1001))
for i in range(1001):
    quad, error = funI(x[i], 0.0)
    value[0,i] = quad
    print(i, x[i], quad)
    quad, error = funI(x[i], np.pi/2.0)
    value[1,i] = quad

np.savez('funI.npz', x, value)