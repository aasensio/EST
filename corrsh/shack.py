import numpy as np

class shackHartmann(object):
    def __init__(self, nPixLens, nLensesX, nLensesY, sigmaNoise, sizeWindowCorr=5):
        self.nPixLens = nPixLens
        self.nLensesX = nLensesX
        self.nLensesY = nLensesY
        self.sigmaNoise = sigmaNoise

        self.lenses = np.zeros((self.nLensesX,self.nLensesY,self.nPixLens,self.nPixLens))

        self.center = [self.nPixLens/2, self.nPixLens/2]
        self.sizeWindowCorr = sizeWindowCorr

        u = np.fft.fftshift(np.fft.fftfreq(self.nPixLens))
        v = np.fft.fftshift(np.fft.fftfreq(self.nPixLens))
        self.fx, self.fy = np.meshgrid(u, v)

        x = np.arange(self.nPixLens)
        y = np.arange(self.nPixLens)
        self.x, self.y = np.meshgrid(x, y)

        self.shiftX = np.zeros((self.nLensesX,self.nLensesY))
        self.shiftY = np.zeros((self.nLensesX,self.nLensesY))

    def generateLenses(self, shiftx, shifty):
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):                
                self.lenses[i,j,:,:] = np.exp(-0.05*((self.x - self.center[0] - shiftx[i,j])**2+(self.y - self.center[1] - shifty[i,j])**2))
        self.lenses += self.sigmaNoise * np.random.randn(self.nLensesX,self.nLensesY,self.nPixLens,self.nPixLens)

    def computeReferenceFFT(self, loc):
        self.referenceFFT = np.fft.fft2(self.lenses[loc[0],loc[1],:,:])

    def computeShiftLens(self, i, j):
        xshiftFFT = np.fft.fft2(self.lenses[i,j,:,:])
        corrFFT = self.referenceFFT * np.conj(xshiftFFT)
        modCorr = np.abs(np.fft.fftshift(corrFFT))
        phaseCorr = np.angle(np.fft.fftshift(corrFFT))

        left = int(self.center[0] - self.sizeWindowCorr)
        right = int(self.center[0] + self.sizeWindowCorr + 1)
        top = int(self.center[1] - self.sizeWindowCorr)
        bottom = int(self.center[1] + self.sizeWindowCorr + 1)

        axx = np.sum(modCorr[left:right,top:bottom] * self.fx[left:right,top:bottom]**2)
        ayy = np.sum(modCorr[left:right,top:bottom] * self.fy[left:right,top:bottom]**2)
        axy = np.sum(modCorr[left:right,top:bottom] * self.fx[left:right,top:bottom] * self.fy[left:right,top:bottom])
        bx = np.sum(modCorr[left:right,top:bottom] * self.fx[left:right,top:bottom] * phaseCorr[left:right,top:bottom])
        by = np.sum(modCorr[left:right,top:bottom] * self.fy[left:right,top:bottom] * phaseCorr[left:right,top:bottom])

        x0 = (bx * ayy - by * axy) / (axx * ayy - axy**2) / (2.0*np.pi)
        y0 = (by * axx - bx * axy) / (axx * ayy - axy**2) / (2.0*np.pi)

        return x0, y0

    def computeAllShifts(self):
        reference = [0,0]
        self.computeReferenceFFT(reference)
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):                
                if (i != reference[0] or j != reference[1]):
                    self.shiftX[i,j], self.shiftY[i,j] = self.computeShiftLens(i, j)