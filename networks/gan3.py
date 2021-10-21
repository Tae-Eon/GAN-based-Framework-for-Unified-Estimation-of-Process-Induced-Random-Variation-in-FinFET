import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class gen3(nn.Module):
    def __init__(self, d_noise_num_of_input, gan_hidden_dim, num_of_output, pdrop):
        super(gen3, self).__init__()
        self.fc1 = nn.Linear(d_noise_num_of_input, gan_hidden_dim)
        self.fc2 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc3 = nn.Linear(gan_hidden_dim, num_of_output)
        self.bn1 = nn.BatchNorm1d(gan_hidden_dim)
        self.bn2 = nn.BatchNorm1d(gan_hidden_dim)
        self.drop_layer = nn.Dropout(p=pdrop)
        
        
    def forward(self, noise, x):
        gen_input = torch.cat((noise, x), axis=1)
        r = self.fc1(gen_input)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.drop_layer(r)
        
        r = self.fc2(r)
        r = self.bn2(r)
        r = F.relu(r)
        r = self.drop_layer(r)
        
        r = self.fc3(r)
        return r
        
class dis3(nn.Module):
    def __init__(self, num_of_output, gan_hidden_dim, pdrop):
        super(dis3, self).__init__()
        
        self.fc1 = nn.Linear(num_of_output, gan_hidden_dim)
        self.fc2 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc3 = nn.Linear(gan_hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(gan_hidden_dim)
        self.bn2 = nn.BatchNorm1d(gan_hidden_dim)
        self.drop_layer = nn.Dropout(p=pdrop)
        
    def forward(self, y, x):
        dis_input = torch.cat((y, x), axis=1)
        r = self.fc1(dis_input)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.drop_layer(r)
        
        r = self.fc2(r)
        r = self.bn2(r)
        r = F.relu(r)
        r = self.drop_layer(r)
        
        r = torch.sigmoid(self.fc3(r))
        return r