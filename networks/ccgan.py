import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ccgen(nn.Module):
    def __init__(self, d_noise_num_of_input, layer, gan_hidden_dim, num_of_output):
        super(ccgen, self).__init__()
                       
        self.network = nn.ModuleList([])

        hidden_dims = [ gan_hidden_dim for i in range(layer-1) ]
        
        # linear model
        if layer == 1:
            self.network.append(nn.Linear(d_noise_num_of_input, num_of_output))
        # multi layer model
        else:
            for h_dim in hidden_dims:
                self.network.append(nn.Linear(d_noise_num_of_input, h_dim))
                self.network.append(nn.ReLU(h_dim))
                d_noise_num_of_input = h_dim
            self.network.append(nn.Linear(d_noise_num_of_input, num_of_output))
        
        
    def forward(self, noise, x):
        r = torch.cat((noise, x), axis=1)
        for module in self.network:
            r = module(r)
        
        return r
        
class ccdis(nn.Module):
    def __init__(self, num_of_output, layer, gan_hidden_dim):
        super(ccdis, self).__init__()
        
        self.network = nn.ModuleList([])
        hidden_dims = [ gan_hidden_dim for i in range(layer-1) ]
        
        # linear model
        if layer == 1:
            self.network.append(nn.Linear(num_of_output, 1))
        # multi layer model
        else:
            for h_dim in hidden_dims:
                self.network.append(nn.Linear(num_of_output, h_dim))
                self.network.append(nn.ReLU(h_dim))
                num_of_output = h_dim
            # last layer
            self.network.append(nn.Linear(num_of_output,1))
        
              
    def forward(self, y, x):
        
        r = torch.cat((y, x), axis=1)
        for module in self.network:
            r = module(r)
        r = torch.sigmoid(r)
        
        return r