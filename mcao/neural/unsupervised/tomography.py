import numpy as np
import matplotlib.pyplot as pl
import glob
import projection
import wavefront as wf
import uuid
import pathlib
import scipy.special as sp
from tqdm import tqdm
from ipdb import set_trace as stop

def even(x):
    return x%2 == 0

def nearest(array, value):
    return (np.abs(array-value)).argmin()


class tomography(object):
    """
    This class defines an atmosphere that can be used to generate synthetic MCAO observations
    and also apply different tomography schemes. It will also be useful for training neural networks
    """
    def __init__(self, nStars, nZernike, fov, heights, DTel, wavelength=5000., verbose=True, numericalProjection=True, addPiston=False, npix=100):
        """
        Class instantiation
        
        Args:
            nStars (int): number of directions (stars) used for the MCAO. They are distributed as one sampling the central part of the cone and
                          the rest distributed around the central one
            nZernike (int): maximum number of Zernike coefficients to use
            fov (float): field-of-view [arcsec]
            heights (float): array of heights to be used [km] in which a turbulent layer can be put. One can always leave one empty afterwards
            DTel (float): telescope diameter [m]
            verbose (bool, optional): turn on verbosity
            numericalProjection (bool, optional): use the numerical approach for computing the projection matrix for the footprints
            addPiston (bool, optional): add piston mode
            npix (int, optional) : number of pixel for the wavefront images
        """
        self.nHeight = len(heights)
        self.nStars = nStars
        self.nZernike = nZernike
        self.fov = fov / 206265.0        # in radians
        self.heights = heights * 1e3
        self.DTel = DTel
        self.verbose = verbose
        self.MComputed = False
        self.numericalProjection = numericalProjection
        self.addPiston = addPiston
        self.noll0 = 1
        self.wavelength = wavelength
        self.npix = npix

        if (not self.addPiston):
            self.noll0 = 2

        # Diameter of the metapupils, which depend on the heights and the FOV
        self.DMetapupil = self.DTel + self.heights * self.fov

        self.t = np.zeros((self.nHeight,self.nStars))
        self.beta = np.zeros((self.nHeight,self.nStars))
        self.angle = np.zeros((self.nHeight,self.nStars))

        if (self.verbose):
            print("-------------------------------------------------------------------")
            print(" - Zernike modes: {0}".format(self.nZernike))
            print(" - Number of heights : {0} -> {1} km".format(self.nHeight, self.heights * 1e-3))
            print(" - FOV: {0} arcsec".format(206265.*self.fov))
            print(" - Number of stars : {0}".format(self.nStars))
            print("-------------------------------------------------------------------")

        # Define the position of the line of sights so that we have one central direction and the rest
        # are divided with azimuthal symmetry around
        # t : radial position of the center of the footprint in units of the meta-pupil radius R<1
        # beta : scaling, so that the radius of the meta-pupil and of the footprint are related by beta=R(metapupil)/R(footprint)
        # angle : azimuthal angle of the line of sight
        for i in range(self.nHeight):
            for j in range(self.nStars-1):
                self.t[i,j] = (self.heights[i] * self.fov) / self.DMetapupil[i]
                self.beta[i,j] = self.DMetapupil[i] / self.DTel
                self.angle[i,j] = j * 2.0 * np.pi / (self.nStars - 1.0)
            self.t[i,-1] = 0.0
            self.beta[i,-1] = self.DMetapupil[i] / self.DTel
            self.angle[i,-1] = 0.0
        
        # Check if the projection matrices for this specific configuration exists. If it does, read it from the file
        # If not, compute them
        
        # First create the directory where all matrices will be saved
        p = pathlib.Path('matrices/')
        p.mkdir(parents=True,exist_ok=True)

        if (self.projectionExists() == 0):
            if (self.verbose):
                print("Projection matrix does not exist")
            self.computeProjection()

        self.aStack = {}
        self.a = {}

        # Read cn2 file
        # cn2 = np.loadtxt('cn2.dat')

        # Compute total r0 value in cm
        # self.r0Reference = (0.423 * (2.0 * np.pi / (self.wavelength*1e-10))**2 * integ.trapz(cn2[:,1], x=cn2[:,0]))**(-3.0/5.0) * 1e2

        # for i in range(len(self.heights)):
        #     indFrom = nearest(cn2[:,0] - cn2[0,0], self.heights[i])
        #     indTo = nearest(cn2[:,0] - cn2[0,0], self.heights[i]+500.0)+1


        self.Z = np.zeros((self.nZernike,self.npix,self.npix))        
        for j in range(self.nZernike):
            self.Z[j,:,:], _, _ = wf.zernike(j+self.noll0,npix=self.npix)

        self.Z = self.Z.reshape((self.nZernike,self.npix*self.npix)).T

        self.pupil, _, _ = wf.zernike(0, npix=self.npix)

    def projectionExists(self):
        """
        Check whether a projection matrix exists
        
        Returns:
            bool: True/False
        """
        
        # Go through all matrices and check if the parameters coincide with what we want
        files = glob.glob('matrices/transformationMatrices*.npz')
        for f in files:
            out = np.load(f)
            heights = out['arr_1']
            nStars = out['arr_2']
            nZernike = out['arr_3']
            fov = out['arr_4']
            DTel = out['arr_5']
            ind = np.where(np.in1d(heights, self.heights))[0]
            if (len(ind) == self.nHeight):
                if (nStars == self.nStars and nZernike >= self.nZernike and 
                    fov == self.fov and DTel == self.DTel):
                    self.M = out['arr_0'][0:self.nZernike,0:self.nZernike,ind,:]

                    # We have found a dataset with the matrices we want. Read it.
                    if (self.verbose):
                        print("Projection matrix exists : {0}".format(f))
                        print(" - Zernike modes: {0}".format(self.nZernike))
                        print(" - Number of heights : {0} -> {1} km".format(self.nHeight, self.heights * 1e-3))
                        print(" - FOV: {0} arcsec".format(206265.*self.fov))
                        print(" - Number of stars : {0}".format(self.nStars))
                        self.MComputed = True
                        self.stackProjection()
                    return True
                
        return False

    def plotPupils(self):
        """
        Plot the pupils                
        """
        ncols = int(np.ceil(np.sqrt(self.nHeight)))
        nrows = int(np.ceil(self.nHeight / ncols))
        cmap = pl.get_cmap('tab10')
        pl.close('all')

        f, ax = pl.subplots(ncols=ncols, nrows=nrows, figsize=(2*ncols,2*nrows))
        ax = ax.flatten()        
        for i in range(self.nHeight):
            radiusMetapupil = self.DMetapupil[i] / 2.0
            circle = pl.Circle((0,0), radiusMetapupil, fill=False, linewidth=2, axes=ax[i])
            ax[i].add_artist(circle)
            ax[i].set_xlim([-0.7*self.DMetapupil[i],0.7*self.DMetapupil[i]])
            ax[i].set_ylim([-0.7*self.DMetapupil[i],0.7*self.DMetapupil[i]])
            ax[i].set_title('h={0} km'.format(self.heights[i] / 1e3))
            for j in range(self.nStars):
                radiusCircle = radiusMetapupil / self.beta[i,j]
                xCircle = radiusMetapupil * self.t[i,j] * np.cos(self.angle[i,j])
                yCircle = radiusMetapupil * self.t[i,j] * np.sin(self.angle[i,j])
                circle = pl.Circle((xCircle,yCircle), radiusCircle, fill=False, axes=ax[i], linewidth=2, color=cmap(j/self.nStars))
                ax[i].add_artist(circle)

    def to_wavefront(self, a):
        tmp = self.Z @ a
        return tmp.reshape((self.npix,self.npix))

    def plot_metapupil(self, index_height):
        """
        Plot the pupils                
        """
        self.generateTurbulentZernikesKolmogorov(5.0)
        
        cmap = pl.get_cmap('tab10')
        pl.close('all')

        f, ax = pl.subplots(ncols=self.nStars, nrows=2, figsize=(19,6))
        
        metapupil = self.to_wavefront(self.a['Original'][:,index_height])

        for j in range(self.nStars):
                    
            # M = projection.zernikeProjectionMatrixNumerical(self.nZernike, self.beta[index_height,j], self.t[index_height,j], self.angle[index_height,j], includePiston=self.addPiston, radius=128)
            # beta = M @ self.a['Original'][:,index_height]

            beta = self.M[:,:,index_height,j] @ self.a['Original'][:,index_height]

            footprint = self.to_wavefront(beta)

            radiusMetapupil = self.DMetapupil[index_height] / 2.0
            radiusCircle = radiusMetapupil / self.beta[index_height,j]
            xCircle = radiusMetapupil * self.t[index_height,j] * np.cos(-self.angle[index_height,j])
            yCircle = radiusMetapupil * self.t[index_height,j] * np.sin(-self.angle[index_height,j])
            circle = pl.Circle((xCircle,yCircle), radiusCircle, fill=False, axes=ax[0,j], linewidth=2)

            ax[0,j].imshow(metapupil, cmap=pl.cm.jet, vmin=-1, vmax=1, extent=[-radiusMetapupil,radiusMetapupil,-radiusMetapupil,radiusMetapupil])
            ax[0,j].add_artist(circle)
            ax[0,j].set_title('Angle: {0:.2f}'.format(self.angle[index_height,j]))

            ax[1,j].imshow(footprint, cmap=pl.cm.jet, vmin=-1, vmax=1, extent=[-radiusCircle,radiusCircle,-radiusCircle,radiusCircle])

        pl.show()
        
    def computeProjection(self):
        """
        Compute the projection matrix for the heights and number of stars defined
        
        Returns:
            None
        """
        if (not self.MComputed):
            self.M = np.zeros((self.nZernike,self.nZernike,self.nHeight,self.nStars))
            for i in tqdm(range(self.nHeight), desc='Height'):                
                for j in tqdm(range(self.nStars), desc='Stars'):                    
                    if (self.numericalProjection):
                        self.M[:,:,i,j] = projection.zernikeProjectionMatrixNumerical(self.nZernike, self.beta[i,j], self.t[i,j], self.angle[i,j], verbose=True, radius=128, includePiston=self.addPiston)
                    else:
                        self.M[:,:,i,j] = projection.zernikeProjectionMatrix(self.nZernike, self.beta[i,j], self.t[i,j], self.angle[i,j], verbose=True, includePiston=self.addPiston)
            np.savez('matrices/transformationMatrices_{0}.npz'.format(uuid.uuid4()), self.M, self.heights, self.nStars, self.nZernike, self.fov, self.DTel)
            self.stackProjection()

    def stackProjection(self):
        """
        Stack the projection matrix to take all heights into account. This facilitates later calculations because
        we can use matrix operations. All Zernike coefficients will be stacked one after the other for all
        metapupils. When multiplied by the matrix, it will make the transformation to the footprints
        
        Returns:
            None
        """
        self.MStack = np.zeros((self.nZernike*self.nStars, self.nZernike*self.nHeight))
        for i in range(self.nHeight):
            for j in range(self.nStars):
                left = i*self.nZernike
                right = (i+1)*self.nZernike
                up = j*self.nZernike
                down = (j+1)*self.nZernike
                self.MStack[up:down,left:right] = self.M[:,:,i,j]

    def generateTurbulentZernikesKolmogorov(self, r0, keepOnly=None):
        """
        A utility to generate the random Zernike coefficients in the metapupils. It uses
        the covariance matrix for the Zernike coefficients for a given value of r0 using Kolmogorov statistics
        
        Args:
            r0 (float): Fried radius [m]
        
        Returns:
            None
        """
        self.covariance = np.zeros((self.nZernike,self.nZernike))
        for i in range(self.nZernike):
            ni, mi = wf.nollIndices(i+self.noll0)
            for j in range(self.nZernike):
                nj, mj = wf.nollIndices(j+self.noll0)
                if (even(i - j)):
                    if (mi == mj):
                        phase = (-1.0)**(0.5*(ni+nj-2*mi))
                        t1 = np.sqrt((ni+1)*(nj+1)) * np.pi**(8.0/3.0) * 0.0072 * (self.DTel / r0)**(5.0/3.0)
                        t2 = sp.gamma(14./3.0) * sp.gamma(0.5*(ni+nj-5.0/3.0))
                        t3 = sp.gamma(0.5*(ni-nj+17.0/3.0)) * sp.gamma(0.5*(nj-ni+17.0/3.0)) * sp.gamma(0.5*(ni+nj+23.0/3.0))
                        self.covariance[i,j] = phase * t1 * t2 / t3

        self.a['Original'] = np.random.multivariate_normal(np.zeros(self.nZernike), self.covariance, size=(self.nHeight)).T

        # Since we might be using projection matrices that contain more heights than 
        # Keep only the heights that we want
        if (keepOnly != None):
            for i in range(self.nHeight):
                if (self.heights[i]/1e3 not in keepOnly):
                    self.a['Original'][:,i] = 0.0

        self.aStack['Original'] = self.a['Original'].T.flatten()


    def generateTurbulentZernikesVonKarman(self, r0, L0):
        """Generate the covariance matrix for the Zernike coefficients for a given value of r0 using von Karman statistics
        
        Args:
            r0 (float): Fried radius [m]
            L0 (float): outer scale [m]
        
        Returns:
            None
        """
        self.covariance = np.zeros((self.nZernike,self.nZernike))
        for i in range(self.nZernike):
            ni, mi = wf.nollIndices(i+self.noll0)
            for j in range(self.nZernike):
                nj, mj = wf.nollIndices(j+self.noll0)
                if (even(i - j)):
                    if (mi == mj):
                        phase = (-1.0)**(0.5*(ni+nj-2*mi))
                        t1 = np.sqrt((ni+1)*(nj+1)) * np.pi**(8.0/3.0) * 1.16 * (self.DTel / r0)**(5.0/3.0)

                        for k in range(50):
                            phase2 = (-1.0)**k / np.math.factorial(k) * (np.pi*self.DTel / L0)**(2.0*k+ni+nj-5.0/3.0)
                            t2 = sp.gamma(k+0.5*(3+ni+nj)) * sp.gamma(k+2+0.5*(ni+nj)) * sp.gamma(k+1+0.2*(ni+nj)) * sp.gamma(5./6.-k-0.5*(ni+nj))
                            t3 = sp.gamma(3+k+ni+nj) * sp.gamma(2+k+ni) * sp.gamma(2+k+nj)

                            phase3 = (np.pi*self.DTel / L0)**(2.0*k)
                            t4 = sp.gamma(0.5*(ni+nj)-5./6.-k) * sp.gamma(k+7./3.) * sp.gamma(k+17/6) * sp.gamma(k+11/6)
                            t5 = sp.gamma(0.5*(ni+nj)+23/6.+k) * sp.gamma(0.5*(ni-nj)+17/6.+k) * sp.gamma(0.5*(ni-nj)+17/6.+k)
                            self.covariance[i,j] += phase * t1 * (phase2 * t2/t3 + phase3 * t4/t5)

        self.a['Original'] = np.random.multivariate_normal(np.zeros(self.nZernike), self.covariance, size=(self.nHeight)).T        
        self.aStack['Original'] = self.a['Original'].T.flatten()

    def generateWFS(self):
        """
        This function uses the stacked alpha coefficients for all metapupils and 
        propagates that to the telescope and all directions
        
        Returns:
            float: array of Zernike coefficients measured in all WFS
        """
        return self.MStack @ self.aStack['Original']
        # self.bStack += np.random.normal(loc=0.0, scale=0.05, size=self.bStack.shape)


    def solveSVD(self, b, regularize=False):
        """Solve the tomography using SVD
        
        Args:
            b (TYPE): array of WFS measurements
            regularize (bool, optional): apply Tikhonov regularization using the covariance matrix of the Zernike coefficients
        
        Returns:
            None
        """
        if (regularize):
            invCov = np.linalg.inv(self.covariance)
            matrixList = [invCov for i in range(self.nHeight)]
            cov = spla.block_diag(*matrixList)
            AInv = np.linalg.inv(self.MStack.T@self.MStack + cov.T@cov)
        else:
            AInv = np.linalg.inv(self.MStack.T@self.MStack)

        x = AInv @ self.MStack.T @ b
        
        self.aStack['SVD'] = x
        self.a['SVD'] = self.aStack['SVD'].reshape((self.nHeight,self.nZernike)).T


if (__name__ == '__main__'):
    np.random.seed(123)
    nStars = 7
    nZernike = 30
    fov = 60.0
    DTel = 4.0

    # Compute the transformation matrices for all heights and directions. This takes some time and 
    # could be easily paralellized using MPI. For instance, 15 heights, 30 Zernike modes and 7 directions
    # took 8 min in my computer
    heights = np.arange(15)
    mcao = tomography(nStars, nZernike, fov, heights, DTel, addPiston=True)
    mcao.plot_metapupil(14)