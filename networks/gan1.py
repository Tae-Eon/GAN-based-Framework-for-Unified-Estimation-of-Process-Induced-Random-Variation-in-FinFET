import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class gen1(nn.Module):
    def __init__(self, d_noise_num_of_input, layer, gan_hidden_dim, num_of_output):
        super(gen1, self).__init__()
        
        self.layer = layer
        print('layer', self.layer)
        
               
        self.fc1 = nn.Linear(d_noise_num_of_input, gan_hidden_dim)
        self.hidden = []
        for k in range(self.layer):
            self.hidden.append(nn.Linear(gan_hidden_dim, gan_hidden_dim))
           
        self.out = nn.Linear(gan_hidden_dim, num_of_output)
        
    def forward(self, noise, x):
        r = self.fc1(torch.cat((noise, x), axis=1))
        for layer in self.hidden:
            r = F.relu(layer(r))
            
        
        r = self.out(r)
        
        return r
        
class dis1(nn.Module):
    def __init__(self, num_of_output, layer, gan_hidden_dim):
        super(dis1, self).__init__()
        
        self.layer = layer
        
        self.fc1 = nn.Linear(num_of_output, gan_hidden_dim)
        
        self.hidden = []
        for k in range(self.layer):
            self.hidden.append(nn.Linear(gan_hidden_dim, gan_hidden_dim))
       
        self.out = nn.Linear(gan_hidden_dim, 1)
        
    def forward(self, y, x):
        r = self.fc1(torch.cat((y, x), axis=1))
        
        for layer in self.hidden:
            r = F.relu(layer(r))
        
        r = torch.sigmoid(self.out(r))
        
        
        return r