import numpy as np
import matplotlib.pyplot as pl
import shack
from ipdb import set_trace as stop

D = 1.0
r0 = 1.0
nModes = 10
sigmaNoise = 0.01
nLensesX = 15
nLensesY = 15
nPixLens = 20

nSamples = 10
sh = shack.shackHartmann(nPixLens, nLensesX, nLensesY, nModes, sigmaNoise, sizeWindowCorr=8)
sh.generateWavefront(D/r0)
sh.generateShifts()
sh.generateLenses()
sh.computeLensCorrelations()
original = sh.coefficients
inferred = sh.solveWavefront(correct=original)
inferredClassical = sh.solveWavefrontClassical()

print(original[0:3], inferred[0:3], inferredClassical[0:3])
sh.draw()