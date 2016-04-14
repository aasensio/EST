import numpy as np
import matplotlib.pyplot as pl
import pyiacsun.optics as optics
from ipdb import set_trace as stop

class shackHartmann(object):
    def __init__(self, nPixLens, nLensesX, nLensesY, nModes, sigmaNoise, sizeWindowCorr=5):
        self.nPixLens = nPixLens
        self.nLensesX = nLensesX
        self.nLensesY = nLensesY
        self.sigmaNoise = sigmaNoise
        self.nModes = nModes
        self.nPixels = self.nPixLens * self.nLensesX

        self.radius = self.nPixLens * self.nLensesX / 2.0

        self.lenses = {'image': np.zeros((self.nLensesX,self.nLensesY,self.nPixLens,self.nPixLens)),
            'center': np.zeros((self.nLensesX,self.nLensesY,2)), 
            'active': np.empty((self.nLensesX, self.nLensesY)),
            'dZX': np.zeros((self.nLensesX, self.nLensesY, self.nModes)),
            'dZY': np.zeros((self.nLensesX, self.nLensesY, self.nModes)),
            'deltaX': np.zeros((self.nLensesX, self.nLensesY)),
            'deltaY': np.zeros((self.nLensesX, self.nLensesY)),
            'inferredX': np.zeros((self.nLensesX, self.nLensesY)),
            'inferredY': np.zeros((self.nLensesX, self.nLensesY))}
        
        self.fullImage = np.zeros((self.nPixels, self.nPixels))

        self.center = np.asarray([self.nPixLens/2, self.nPixLens/2])

        self.globalCenter = np.asarray([self.nPixels/2, self.nPixels/2])

        self.sizeWindowCorr = sizeWindowCorr

        u = np.fft.fftshift(np.fft.fftfreq(self.nPixLens))
        v = np.fft.fftshift(np.fft.fftfreq(self.nPixLens))
        self.fx, self.fy = np.meshgrid(u, v)

        x = np.arange(self.nPixLens)
        y = np.arange(self.nPixLens)
        self.x, self.y = np.meshgrid(x, y)

        self.referenceLens = np.asarray([int(self.nLensesX/2), int(self.nLensesY/2)])

# Generate the Zernike machine
        self.ZMachine = optics.zernikeMachine(npix=self.nPixels)

        self.wavefront = np.zeros((self.nPixels,self.nPixels))

# Compute the center of the subapertures
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):                
                self.lenses['center'][i,j,:] = np.asarray([-self.radius + i*self.nPixLens + self.nPixLens/2, 
                    -self.radius + j*nPixLens + self.nPixLens/2])

# Compute the values of the gradient of the Zernike polynomials at each subaperture center
# This is used for developing the shifts given the wavefront
        for k in range(self.nModes):
            dZX, dZY = self.ZMachine.gradZernike(k+2)
            for i in range(self.nLensesX):
                for j in range(self.nLensesY):
                    self.lenses['dZX'][i,j,k] = dZX[self.lenses['center'][i,j,0] + self.globalCenter[0], 
                        self.lenses['center'][i,j,1] + self.globalCenter[1]]
                    self.lenses['dZY'][i,j,k] = dZY[self.lenses['center'][i,j,0] + self.globalCenter[0], 
                        self.lenses['center'][i,j,1] + self.globalCenter[1]]

        self.nActive = self.countActiveLenses()

        self.correlations = {'axx': np.zeros((self.nLensesX, self.nLensesY)),
            'axy': np.zeros((self.nLensesX, self.nLensesY)),
            'ayy': np.zeros((self.nLensesX, self.nLensesY)),
            'bx': np.zeros((self.nLensesX, self.nLensesY)),
            'by': np.zeros((self.nLensesX, self.nLensesY))}
    
    def generateLenses(self):
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):                
                self.lenses['image'][i,j,:,:] = 1.0 - 0.05*np.exp(-0.05*((self.x - self.center[0] - self.lenses['deltaX'][i,j])**2
                    + (self.y - self.center[1] - self.lenses['deltaY'][i,j])**2))

                if (self.testActiveLens(i,j)):
                    self.fullImage[self.lenses['center'][i,j,0]-self.nPixLens/2+ self.globalCenter[0]:self.lenses['center'][i,j,0]+self.nPixLens/2+ self.globalCenter[0],
                        self.lenses['center'][i,j,1]-self.nPixLens/2+ self.globalCenter[1]:self.lenses['center'][i,j,1]+self.nPixLens/2+ self.globalCenter[1]] = self.lenses['image'][i,j,:,:]
        
        self.lenses['image'] += self.sigmaNoise * np.random.randn(self.nLensesX,self.nLensesY,self.nPixLens,self.nPixLens)

    def generateWavefront(self, D_over_r0):
        self.coefficients = self.ZMachine.generateTurbulentZernikesKolmogorov(D_over_r0, self.nModes, firstMode=2)
        self.wavefront = 0.0
        for k in range(self.nModes):
            Z = self.ZMachine.zernikeNoll(k+2)
            self.wavefront += self.coefficients[k] * Z

    def generateShifts(self):
        self.lenses['deltaX'] = np.sum(self.lenses['dZX'] * self.coefficients[None,None,:], axis=2)
        self.lenses['deltaY'] = np.sum(self.lenses['dZY'] * self.coefficients[None,None,:], axis=2)
        self.lenses['deltaX'][self.referenceLens[0],self.referenceLens[1]] = 0.0
        self.lenses['deltaY'][self.referenceLens[0],self.referenceLens[1]] = 0.0

    def computeReferenceFFT(self):
        self.referenceFFT = np.fft.fft2(self.lenses['image'][self.referenceLens[0],self.referenceLens[1],:,:])

    def computeLensCorrelations(self):
        left = int(self.center[0] - self.sizeWindowCorr)
        right = int(self.center[0] + self.sizeWindowCorr + 1)
        top = int(self.center[1] - self.sizeWindowCorr)
        bottom = int(self.center[1] + self.sizeWindowCorr + 1)

        self.computeReferenceFFT()

        for i in range(self.nLensesX):
            for j in range(self.nLensesY):
                if (self.testActiveLens(i,j)):
                    if (self.referenceLens[0] != i and self.referenceLens[1] != j):
                        xshiftFFT = np.fft.fft2(self.lenses['image'][i,j,:,:])
                        corrFFT = np.fft.fftshift(self.referenceFFT * np.conj(xshiftFFT))
                        modCorr = np.abs(corrFFT)
                        phaseCorr = np.angle(corrFFT)

                        self.correlations['axx'][i,j] = np.sum(modCorr[left:right,top:bottom] * self.fx[left:right,top:bottom]**2)
                        self.correlations['ayy'][i,j] = np.sum(modCorr[left:right,top:bottom] * self.fy[left:right,top:bottom]**2)
                        self.correlations['axy'][i,j] = np.sum(modCorr[left:right,top:bottom] * self.fx[left:right,top:bottom] * self.fy[left:right,top:bottom])
                        self.correlations['bx'][i,j] = np.sum(modCorr[left:right,top:bottom] * self.fx[left:right,top:bottom] * phaseCorr[left:right,top:bottom])
                        self.correlations['by'][i,j] = np.sum(modCorr[left:right,top:bottom] * self.fy[left:right,top:bottom] * phaseCorr[left:right,top:bottom])

                        den = 2.0 * np.pi * (self.correlations['axx'][i,j] * self.correlations['ayy'][i,j] - self.correlations['axy'][i,j]**2)
                        num = self.correlations['bx'][i,j] * self.correlations['ayy'][i,j] - self.correlations['by'][i,j] * self.correlations['axy'][i,j]
                        self.lenses['inferredX'][i,j] = num / den

                        num = self.correlations['by'][i,j] * self.correlations['axx'][i,j] - self.correlations['bx'][i,j] * self.correlations['axy'][i,j]
                        self.lenses['inferredY'][i,j] = num / den
        return

    def solveWavefront(self, correct=None):
        A = np.zeros((self.nModes,self.nModes))
        b = np.zeros(self.nModes)

        b = np.sum(self.lenses['dZX'][self.xActive,self.yActive,:]*self.correlations['bx'][self.xActive,self.yActive,None], axis=0)
        b += np.sum(self.lenses['dZY'][self.xActive,self.yActive,:]*self.correlations['by'][self.xActive,self.yActive,None], axis=0)
        b /= (2.0*np.pi)

        A = np.einsum('ik,ij,i', self.lenses['dZX'][self.xActive,self.yActive,:], 
            self.lenses['dZX'][self.xActive,self.yActive,:],
            self.correlations['axx'][self.xActive,self.yActive])

        A += np.einsum('ik,ij,i', self.lenses['dZX'][self.xActive,self.yActive,:], 
            self.lenses['dZY'][self.xActive,self.yActive,:],
            self.correlations['axy'][self.xActive,self.yActive])

        A += np.einsum('ik,ij,i', self.lenses['dZY'][self.xActive,self.yActive,:], 
            self.lenses['dZX'][self.xActive,self.yActive,:],
            self.correlations['axy'][self.xActive,self.yActive])

        A += np.einsum('ik,ij,i', self.lenses['dZY'][self.xActive,self.yActive,:], 
            self.lenses['dZY'][self.xActive,self.yActive,:],
            self.correlations['ayy'][self.xActive,self.yActive])

        # U, W, VT = np.linalg.svd(A)
        # invW = 1.0 / W
        # invW[W < 1e-8 * np.max(W)] = 0.0
        # self.inferred = U @ np.diag(invW) @ VT @ b

        self.inferred = np.linalg.pinv(A) @ b

        return self.inferred

    def solveWavefrontClassical(self):
        A = np.vstack([self.lenses['dZX'][self.xActive,self.yActive,:],self.lenses['dZY'][self.xActive,self.yActive,:]])
        b = np.hstack([self.lenses['inferredX'][self.xActive,self.yActive], self.lenses['inferredY'][self.xActive,self.yActive]])

        self.inferredClassical = np.linalg.pinv(A) @ b

        return self.inferredClassical

    def countActiveLenses(self):
        self.nActive = 0
        xActive = []
        yActive = []
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):
                if (self.testActiveLens(i,j)):
                    if (self.referenceLens[0] != i and self.referenceLens[1] != j):
                        self.nActive += 1
                        xActive.append(i)
                        yActive.append(j) 
        self.xActive = np.asarray(xActive)
        self.yActive = np.asarray(yActive)

        return

    def testActiveLens(self, i, j):
        corner = np.zeros((4,2))
        corner[0,:] = self.lenses['center'][i,j,:] + np.asarray([self.nPixLens/2,self.nPixLens/2])
        corner[1,:] = self.lenses['center'][i,j,:] + np.asarray([-self.nPixLens/2,self.nPixLens/2])
        corner[2,:] = self.lenses['center'][i,j,:] + np.asarray([self.nPixLens/2,-self.nPixLens/2])
        corner[3,:] = self.lenses['center'][i,j,:] + np.asarray([-self.nPixLens/2,-self.nPixLens/2])

        distance = np.sum(corner**2, axis=1)
        if np.any(distance > self.radius**2):
            self.lenses['active'][i,j] = False
            return False
        self.lenses['active'][i,j] = True
        return True

    def draw(self):
        pl.close('all')
        f, ax = pl.subplots(nrows=2, ncols=2, figsize=(16,12))
        ax = ax.flatten()
        image = ax[0].imshow(self.wavefront, extent=(-self.nPixels/2,self.nPixels/2,-self.nPixels/2,self.nPixels/2), origin='lower', cmap=pl.cm.jet)
        pl.colorbar(image, ax=ax[0])
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):
                lens = pl.Rectangle((-self.radius + i*self.nPixLens, -self.radius + j*self.nPixLens), self.nPixLens, self.nPixLens, fill=False, axes=ax[0])
                if (self.testActiveLens(i,j)):
                    if (self.referenceLens[0] == i and self.referenceLens[1] == j):
                        ax[0].plot(self.lenses['center'][i,j,0], self.lenses['center'][i,j,1], 'o', color='red')
                        ax[0].plot(self.lenses['center'][i,j,0] + self.lenses['deltaX'][i,j], 
                            self.lenses['center'][i,j,1] + self.lenses['deltaY'][i,j], 'o', color='green')
                        ax[0].arrow(self.lenses['center'][i,j,0], self.lenses['center'][i,j,1], 
                            self.lenses['deltaX'][i,j], self.lenses['deltaY'][i,j])
                    else:
                        ax[0].plot(self.lenses['center'][i,j,0], self.lenses['center'][i,j,1], 'o', color='black')
                        ax[0].plot(self.lenses['center'][i,j,0] + self.lenses['deltaX'][i,j], 
                            self.lenses['center'][i,j,1] + self.lenses['deltaY'][i,j], 'o', color='green')
                        ax[0].arrow(self.lenses['center'][i,j,0], self.lenses['center'][i,j,1], 
                            self.lenses['deltaX'][i,j], self.lenses['deltaY'][i,j])
                else:
                    ax[0].plot(self.lenses['center'][i,j,0], self.lenses['center'][i,j,1], 'x', color='black')
                ax[0].add_artist(lens)

        circle = pl.Circle((0,0), self.radius, fill=False, linewidth=2, axes=ax[0])
        ax[0].add_artist(circle)
        ax[0].set_xlim([-self.radius, self.radius])
        ax[0].set_ylim([-self.radius, self.radius])


        ax[1].imshow(self.fullImage, extent=(-self.nPixels/2,self.nPixels/2,-self.nPixels/2,self.nPixels/2), origin='lower', cmap=pl.cm.jet)
        for i in range(self.nLensesX):
            for j in range(self.nLensesY):
                lens = pl.Rectangle((-self.radius + i*self.nPixLens, -self.radius + j*self.nPixLens), self.nPixLens, self.nPixLens, fill=False, axes=ax[1])
                if (not self.testActiveLens(i,j)):
                    ax[1].plot(self.lenses['center'][i,j,0], self.lenses['center'][i,j,1], 'x', color='black')
                ax[1].add_artist(lens)
        circle = pl.Circle((0,0), self.radius, fill=False, linewidth=2, axes=ax[1])
        ax[1].add_artist(circle)
        ax[1].set_xlim([-self.radius, self.radius])
        ax[1].set_ylim([-self.radius, self.radius])

        self.wavefrontInferred = 0.0
        for k in range(self.nModes):
            Z = self.ZMachine.zernikeNoll(k+2)
            self.wavefrontInferred += self.inferred[k] * Z
        image = ax[2].imshow(self.wavefrontInferred, extent=(-self.nPixels/2,self.nPixels/2,-self.nPixels/2,self.nPixels/2), origin='lower', cmap=pl.cm.jet)
        pl.colorbar(image, ax=ax[2])

        image = ax[3].imshow(self.wavefront - self.wavefrontInferred, extent=(-self.nPixels/2,self.nPixels/2,-self.nPixels/2,self.nPixels/2), origin='lower', cmap=pl.cm.jet)
        pl.colorbar(image, ax=ax[3])