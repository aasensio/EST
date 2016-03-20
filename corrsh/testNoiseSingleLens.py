import numpy as np
import matplotlib.pyplot as pl
import shack

sh = shack.shackHartmann(50, 2, 1, 0.01)
shiftx = np.asarray([[0.0],[3.0]])
shifty = np.asarray([[0.0],[0.0]])
x = np.zeros(500)
for i in range(500):
    sh.generateLenses(shiftx, shifty)
    sh.computeAllShifts()
    x[i] = sh.shiftX[1,0]

pl.plot(x)