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
from tqdm import tqdm
from ipdb import set_trace as stop

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class dataset(torch.utils.data.Dataset):
    def __init__(self, n_training, n_layers_turbulence, n_total_heights, n_directions, n_zernike, fov, heights, DTel):
        super(dataset, self).__init__()
        
        self.n_training = n_training
        self.n_zernike = n_zernike
        self.n_layers_turbulence = n_layers_turbulence        
        self.n_directions = n_directions
        self.fov = fov
        self.heights = heights
        self.n_total_heights = len(self.heights)
        self.DTel = DTel

        self.mcao = tomography.tomography(self.n_directions, self.n_zernike, self.fov, self.heights, self.DTel, addPiston=True, npix=100)

    def __getitem__(self, index):

        alpha = np.random.randn(self.n_zernike, self.n_layers_turbulence)
        heights = np.random.permutation(self.heights)[0:self.n_layers_turbulence]
        beta = np.zeros((self.n_zernike, self.n_directions))

        for i in range(self.n_directions):
            for j in range(self.n_layers_turbulence):        
                beta[:,i] += self.mcao.M[:,:,heights[j],i] @ alpha[:,j]

        return beta.astype('float32')

    def __len__(self):
        return self.n_training


class mcao_neural(object):
    def __init__(self, batch_size, n_layers_turbulence, n_training, n_validation, lr):

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.batch_size = batch_size
        self.lr = lr

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

        dset = dataset(self.n_training, self.n_layers_turbulence, self.n_total_heights, self.n_directions, self.n_zernike, self.fov, self.heights, self.DTel)
        self.train_loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True, **kwargs)

        dset = dataset(self.n_validation, self.n_layers_turbulence, self.n_total_heights, self.n_directions, self.n_zernike, self.fov, self.heights, self.DTel)
        self.val_loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True, **kwargs)

        
        self.loss_fn = nn.MSELoss().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def optimize(self, n_epochs):
        self.loss = []
        self.loss_val = []
        best_loss = -1e10
        self.n_epochs = n_epochs

        trainF = open('{0}.loss.csv'.format(self.out_name), 'w')

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):
            self.train(epoch)
            self.test(epoch)

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = min(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))


        trainF.close()

    def train(self, epoch):
        self.model.train()

        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        n = 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (beta_turb) in enumerate(t):
            beta_turb = beta_turb.to(self.device)
            
            self.optimizer.zero_grad()

            d = self.model(beta_turb)

            d_flatten = d.view(-1,self.n_zernike)

            beta = beta_turb * 0.0

            for i in range(self.n_directions):
                for j in range(self.n_layers_DMs):
                    mat = self.M[:,:,self.index_h_DMs[j],i]
                    tmp = mat @ d[:,j,:].t()
                    beta[:,:,i] += tmp.t()

            loss = self.loss_fn(beta_turb, beta)
            
            loss_avg += (loss.item() - loss_avg) / n            
            n += 1

            loss.backward()

            self.optimizer.step()
            
            t.set_postfix(loss=loss.item(), loss_avg=loss_avg)
            
        self.loss.append(loss_avg)

    def test(self, epoch):
        self.model.eval()

        t = tqdm(self.val_loader)
        loss_avg = 0.0
        n = 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (beta_turb) in enumerate(t):
            beta_turb = beta_turb.to(self.device)
            
            d = self.model(beta_turb)

            beta = beta_turb * 0.0

            for i in range(self.n_directions):
                for j in range(self.n_layers_DMs):
                    mat = self.M[:,:,self.index_h_DMs[j],i]
                    tmp = mat @ d[:,j,:].t()
                    beta[:,:,i] += tmp.t()

            loss = self.loss_fn(beta_turb, beta)
            
            loss_avg += (loss.item() - loss_avg) / n            
            n += 1
                        
            t.set_postfix(loss=loss.item(), loss_avg=loss_avg)
            
        self.loss_val.append(loss_avg)


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    
    parsed = vars(parser.parse_args())

    mcao_network = mcao_neural(batch_size=64, n_layers_turbulence=2, n_training=1000*64, n_validation=50*64, lr=parsed['lr'])

    mcao_network.optimize(n_epochs=10)