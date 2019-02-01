import numpy as np
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
import seaborn as sns

def plotOne():
    out = np.load('results/oneLayer.npz')
    original = np.asarray(out['arr_0'])
    reconstructed = np.asarray(out['arr_1'])

    pl.close('all')
    f, ax = pl.subplots(nrows=2, ncols=3, figsize=(12,8))
    ax = ax.flatten()

    testHeights = [0, 3, 10, 16, 20, 25]

    n = original.shape[1]
    h = np.linspace(0,32,n)

    for i in range(6):
        ax[i].plot(h, original[i,:,0] - reconstructed[i,:,0])
        # ax[i].plot(h, reconstructed[i,:,0])
        ax[i].set_xlabel('Height [km]')
        ax[i].set_ylabel('Residual')
        ax[i].set_title('{0} km'.format(testHeights[i]))
        ax[i].set_xlim([0,32])
        ax[i].set_ylim([-0.2,0.2])

    pl.tight_layout()

    stop()
    # pl.savefig('figs/oneLayer.png')

def plotTwo():
    f, ax = pl.subplots(nrows=4, ncols=4, figsize=(12,8))
    out = np.load('results/twoLayer.npz')
    original = np.asarray(out['arr_0'])
    reconstructed = np.asarray(out['arr_1'])
    testHeights = [0, 3, 16, 25]
    for i in range(4):
        for j in range(4):
            ax[i,j].plot(original[i,j,:])
            ax[i,j].plot(reconstructed[i,j,:])
            ax[i,j].set_title('i={0} j={1}'.format(i,j))
    pl.tight_layout()
    
# plotOne()

plotTwo()
