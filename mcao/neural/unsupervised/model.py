import torch
import torch.nn as nn
import torch.utils.data
from ipdb import set_trace as stop

class network(nn.Module):
    def __init__(self, n_zernike, n_directions, n_layers_DMs, batch_size):
        super(network, self).__init__()

        self.n_zernike = n_zernike
        self.n_directions = n_directions
        self.n_layers_DMs = n_layers_DMs
        self.batch_size = batch_size
        
        self.layer1 = nn.Linear(n_zernike*n_directions, 256)
        self.layer2 = nn.Linear(256, n_zernike*n_layers_DMs)
        self.relu = nn.ReLU()                
        
    def forward(self, x):

        out = self.layer1(x.view(self.batch_size, -1))
        out = self.relu(out)
        out = self.layer2(out)        
        
        return out.view(self.batch_size, self.n_layers_DMs, self.n_zernike)