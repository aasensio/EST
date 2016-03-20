import numpy as np
import matplotlib.pyplot as pl
import pyiacsun as ps
from ipdb import set_trace as stop

class covariance(object):
    """docstring for covariance"""
    def __init__(self):
        out = np.load('funI.npz')
        self.funS, self.funI = out['arr_0'], out['arr_1']

        self.nx = 40
        self.ny = 40
        alpha = np.linspace(0.0, 45.5, self.nx) / 206265.0
        ss = np.linspace(0.0, 0.98, self.ny)
        a, s = np.meshgrid(alpha, ss)
        self.a = a.flatten()
        self.s = s.flatten()
        self.diameter = 0.098

    def funI0(self, s):
        return np.interp(np.abs(s), self.funS, self.funI[0,:])

    def reshape(self, x):
        return x.reshape((self.nx, self.ny))

    def Fx(self):        
        
        h = np.asarray([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 9.5, 16.0, 30.0]) * 1e3
        nh = len(h)

        x, w = ps.util.gauss_weights((0.0, 30.0), nh)

        pl.close('all')

        f, ax = pl.subplots(ncols=2, nrows=5, figsize=(10,12))
        ax = ax.flatten()
        for i in range(10):
            Deff = np.max([5.5 / 206265.0 * h[i], self.diameter])
            print(i,Deff)
            t1 = self.reshape(self.funI0((self.a * h[i] - self.s) / Deff))
            t2 = self.reshape(self.funI0((self.a * h[i] + self.s) / Deff))
            t3 = self.reshape(self.funI0(self.a * h[i] / Deff))
            res = 0.5 * (t1 + t2) - t3
            ax[9-i].imshow(res, cmap=pl.cm.gray, origin='lower')

        stop()

out = covariance()
out.Fx()