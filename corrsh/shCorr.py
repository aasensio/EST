import numpy as np
import matplotlib.pyplot as pl
import shack
import seaborn as sns
from ipdb import set_trace as stop


D = 1.0
r0 = 0.3
nModes = 10
sigmaNoise = 0.01
nLensesX = 5
nLensesY = 5
nPixLens = 50

nSamples = 10
sh = shack.shackHartmann(nPixLens, nLensesX, nLensesY, nModes, sigmaNoise, sizeWindowCorr=3)
original = np.zeros((nModes, nSamples))
inferred = np.zeros((nModes, nSamples))
inferredClassical = np.zeros((nModes, nSamples))
for i in range(nSamples):
    sh.generateWavefront(D/r0)
    sh.generateShifts()
    sh.generateLenses()
    sh.computeLensCorrelations()
    original[:,i] = sh.coefficients
    inferred[:,i] = sh.solveWavefront(correct=original[:,i])
    inferredClassical[:,i] = sh.solveWavefrontClassical()

    print(original[0:3,i], inferred[0:3,i], inferredClassical[0:3,i])
    # print(np.median(sh.lenses['deltaX']), np.median(sh.lenses['deltaY']))
    # print(np.median(sh.lenses['inferredX']), np.median(sh.lenses['inferredY']))
    # sh.draw()

ncol = np.ceil(np.sqrt(nModes))
nrow = np.ceil(nModes / ncol)
f, ax = pl.subplots(nrows=int(nrow), ncols=int(ncol), figsize=(12,6))
ax = ax.flatten()
cmap = sns.color_palette()

for i in range(nModes):
    ax[i].plot(original[i,:], color=cmap[0], label='Original')
    ax[i].plot(inferred[i,:], color=cmap[1], label='Inferred')
    ax[i].plot(inferredClassical[i,:], color=cmap[2], label='Classical')

ax[0].legend()

# #sh.draw()