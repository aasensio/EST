import numpy as np
import matplotlib.pyplot as pl
import tomography
import torch
import torch.nn as nn
import torch.utils.data
import argparse
import model
import time
import shutil
import glob
import os
import scipy.special as sp
from tqdm import tqdm
from ipdb import set_trace as stop

def noll_indices(j):
  narr = np.arange(40)
  jmax = (narr+1)*(narr+2)/2
  wh = np.where(j <= jmax)
  n = wh[0][0]
  mprime = j - n*(n+1)/2
  if ((n % 2) == 0):
    m = 2*int(np.floor(mprime/2))
  else:
    m = 1 + 2*int(np.floor((mprime-1)/2))

  if ((j % 2) != 0):
    m *= -1

  return n, m

def _even(x):
    return x%2 == 0

def _zernike_parity(j, jp):
    return _even(j-jp)

class mcao_neural(object):
    def __init__(self, checkpoint, batch_size, n_layers_turbulence, n_training, n_validation):

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.batch_size = batch_size

        if (checkpoint is None):
            files = glob.glob('trained/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)
            
        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)

        # Define the properties of the telescope, FOV, etc.
        self.n_directions = 7
        self.n_zernike = 30
        self.fov = 60.0
        self.DTel = 4.0
        self.npix = 100
        self.n_layers_turbulence = n_layers_turbulence
        self.n_training = n_training
        self.n_validation = n_validation

        # Define all potential heights that can harbor turbulence or DMs
        self.heights = np.arange(15)
        self.n_total_heights = len(self.heights)

        # Instantiate the tomography class. This reads the projection matrix and defines
        # some useful functions
        self.mcao = tomography.tomography(self.n_directions, self.n_zernike, self.fov, self.heights, self.DTel, addPiston=True, npix=self.npix)

        self.M = torch.from_numpy(self.mcao.M.astype('float32')).to(self.device)

# Indices of the height vector on which we have a DM
        self.index_h_DMs = [0, 11]
        self.n_layers_DMs = len(self.index_h_DMs)

        torch.backends.cudnn.benchmark = True

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'trained/{0}'.format(current_time)

        self.model = model.network(self.n_zernike, self.n_directions, self.n_layers_DMs, self.batch_size).to(self.device)

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])        
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

    def generate_turbulence(self, r0):
        
        covariance = np.zeros((self.n_zernike,self.n_zernike))
        for j in range(self.n_zernike):
            n, m = noll_indices(j+1)
            for jpr in range(self.n_zernike):
                npr, mpr = noll_indices(jpr+1)
                
                deltaz = (m == mpr) and (_zernike_parity(j, jpr) or m == 0)
                
                if (deltaz):                
                    phase = (-1.0)**(0.5*(n+npr-2*m))
                    t1 = np.sqrt((n+1)*(npr+1)) 
                    t2 = sp.gamma(14./3.0) * sp.gamma(11./6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)

                    Kzz = t2 * t1 * phase
                    
                    t1 = sp.gamma(0.5*(n+npr-5.0/3.0))
                    t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                    covariance[j,jpr] = Kzz * t1 / t2
        
                        
        covariance[0,0] = 1.0
        covariance[0,:] = 0.0
        covariance[:,0] = 0.0

        return np.random.multivariate_normal(np.zeros(self.n_zernike), covariance)

    def validate(self):
        self.model.eval()

        wfs = np.zeros((self.n_directions, self.npix, self.npix))
        loss = np.zeros(self.n_directions)
        mask = self.mcao.pupil == 1
        n_pixel = np.sum(mask)

        pl.close('all')
        f, ax = pl.subplots(nrows=3, ncols=3, figsize=(13,8), constrained_layout=True)
        f2, ax2 = pl.subplots(nrows=3, ncols=3, figsize=(13,8), constrained_layout=True)
        ax = ax.flatten()
        ax2 = ax2.flatten()

        with torch.no_grad():

            alpha = np.zeros((self.n_zernike,self.n_layers_turbulence))
            alpha[:,0] = self.generate_turbulence(r0=2.0)
            alpha[:,1] = self.generate_turbulence(r0=2.0)
            alpha = torch.from_numpy(alpha.astype('float32')).to(self.device)

            heights = np.random.permutation(self.heights)[0:self.n_layers_turbulence]

            beta_turb = torch.zeros((1,self.n_zernike, self.n_directions)).to(self.device)

            for i in range(self.n_directions):
                for j in range(self.n_layers_turbulence):                    
                    tmp = self.M[:,:,heights[j],i] @ alpha[:,j]
                    beta_turb[0,:,i] += tmp

            # beta_turb = beta_turb + 1e-1 * torch.normal(mean=torch.ones((1,self.n_zernike, self.n_directions)), std=1.0).to(self.device)
    
            d = self.model(beta_turb)

            beta = beta_turb * 0.0

            for i in range(self.n_directions):
                for j in range(self.n_layers_DMs):
                    mat = self.M[:,:,self.index_h_DMs[j],i]
                    tmp = mat @ d[:,j,:].t()
                    beta[:,:,i] += tmp.t()

            beta = beta_turb - beta
            alpha = alpha.cpu().numpy()
            beta = beta.cpu().numpy()
            beta_turb = beta_turb.cpu().numpy()
            d = d.cpu().numpy()

            for i in range(7):
                wfs[i,:,:] = self.mcao.to_wavefront(beta[0,:,i])
                loss[i] = np.sum(wfs[i,mask]**2) / n_pixel

                im = ax[i].imshow(wfs[i,:,:], cmap=pl.cm.viridis)
                ax[i].set_title('Corrected WF rms={0:.2f}'.format(loss[i]))
                pl.colorbar(im, ax=ax[i])

                wfs[i,:,:] = self.mcao.to_wavefront(beta_turb[0,:,i])
                loss[i] = np.sum(wfs[i,mask]**2) / n_pixel

                im = ax2[i].imshow(wfs[i,:,:], cmap=pl.cm.viridis)        
                ax2[i].set_title('Original WF rms={0:.2f}'.format(loss[i]))
                pl.colorbar(im, ax=ax2[i])

            for i in range(2):
                wfs[i,:,:] = self.mcao.to_wavefront(alpha[:,i])
                im = ax[i+7].imshow(wfs[i,:,:], cmap=pl.cm.viridis)
                ax[i+7].set_title('Turbulence metapupil h={0}'.format(heights[i]))
                pl.colorbar(im, ax=ax[i+7])
                
                wfs[i,:,:] = self.mcao.to_wavefront(d[0,i,:])
                im = ax2[i+7].imshow(wfs[i,:,:], cmap=pl.cm.viridis)
                ax2[i+7].set_title('DMs h={0}'.format(self.index_h_DMs[i]))
                pl.colorbar(im, ax=ax2[i+7])
    
            pl.show()

            f.savefig('original.png', bbox_inches='tight')
            f2.savefig('corrected.png', bbox_inches='tight')
            

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    
    parsed = vars(parser.parse_args())

    mcao_network = mcao_neural(checkpoint=None, batch_size=1, n_layers_turbulence=2, n_training=100*64, n_validation=10*64)

    mcao_network.validate()