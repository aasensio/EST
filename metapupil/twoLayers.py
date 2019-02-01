import numpy as np
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
import seaborn as sns
import tomography

def plotResults(forward, inversion, index):
    f, ax = pl.subplots(nrows=2, ncols=1, figsize=(12,10))
    ax[0].plot(inversion.aStack['SVD'], label='SVD')
    ax[0].plot(forward.aStack['Original'], label='Original')
    ax[1].plot(inversion.aStack['L1'], label='l1')
    ax[1].plot(forward.aStack['Original'], label='Original')
    ax[0].legend()
    ax[1].legend()

np.random.seed(123)
sns.set_style("dark")
nStars = 7
nZernike = 30
fov = 60.0
r0 = 0.15
DTel = 4.0
heights = np.arange(31)
noises = np.asarray([0.00001, 0.05, 0.1])
nNoises = len(noises)

testHeights = [0, 3, 16, 25]
nHeights = len(testHeights)

original = np.zeros((nHeights,nHeights,31*nZernike))
reconstructed = np.zeros((nHeights,nHeights,31*nZernike))

noiseStd = 0.01

forward = tomography.tomography(nStars, nZernike, fov, heights, DTel)
for j in range(nHeights):
    for i in range(nHeights):
        if (testHeights[i] > testHeights[j]):
            forward.generateTurbulentZernikesKolmogorov(r0, layers=[testHeights[i],testHeights[j]], percentages=[75.0, 25.0])
            bMeasured = forward.generateWFS()
            bMeasured += np.random.normal(scale=noiseStd, size=bMeasured.shape)
            
            inversion = tomography.tomography(nStars, nZernike, fov, heights, DTel)
            inversion.generateTurbulentZernikesKolmogorov(r0, layers=[testHeights[i],testHeights[j]], percentages=[75.0, 25.0])
            inversion.solveSVD(bMeasured, regularize=True)

            # inversion.solveFASTA(bMeasured, mu=0.001)

            inversion.solveBSBL(bMeasured, lambdaPar=noiseStd**2)

            # pl.plot(forward.a['Original'].T.flatten())
            # pl.plot(inversion.a['L1'].T.flatten())
            # stop()

            original[i,j,:] = forward.a['Original'].T.flatten()
            reconstructed[i,j,:] = inversion.a['L1'].T.flatten()
        
np.savez('results/twoLayer.npz', original, reconstructed)