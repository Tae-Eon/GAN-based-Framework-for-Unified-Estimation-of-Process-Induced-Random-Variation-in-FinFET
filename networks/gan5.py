import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class gen5(nn.Module):
    def __init__(self, d_noise_num_of_input, gan_hidden_dim, num_of_output):
        super(gen5, self).__init__()
        self.fc1 = nn.Linear(d_noise_num_of_input, gan_hidden_dim)
        self.fc2 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc3 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc4 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc5 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc6 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc7 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc8 = nn.Linear(gan_hidden_dim, num_of_output)
        
    def forward(self, noise, x):
        gen_input = torch.cat((noise, x), axis=1)
        r = F.relu(self.fc1(gen_input))
        r = F.relu(self.fc2(r))
        r = F.relu(self.fc3(r))
        r = F.relu(self.fc4(r))
        r = F.relu(self.fc5(r))
        r = F.relu(self.fc6(r))
        r = F.relu(self.fc7(r))
        r = self.fc8(r)
        
        return r
        
class dis5(nn.Module):
    def __init__(self, num_of_output, gan_hidden_dim, pdrop):
        super(dis5, self).__init__()
        
        self.drop_layer = nn.Dropout(p=pdrop)

        
        self.fc1 = nn.Linear(num_of_output, gan_hidden_dim)
        self.fc2 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc3 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc4 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc5 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc6 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc7 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc8 = nn.Linear(gan_hidden_dim, 1)
        
    def forward(self, y, x):
        dis_input = torch.cat((y, x), axis=1)
        r = self.drop_layer(self.fc1(dis_input))
        r = F.relu(r)
        r = self.drop_layer(self.fc2(r))
        r = F.relu(r)
        r = self.drop_layer(self.fc3(r))
        r = F.relu(r)
        r = self.drop_layer(self.fc4(r))
        r = F.relu(r)
        r = self.drop_layer(self.fc5(r))
        r = F.relu(r)
        r = self.drop_layer(self.fc6(r))
        r = F.relu(r)
        r = self.drop_layer(self.fc7(r))
        r = F.relu(r)
        r = torch.sigmoid(self.fc8(r))
        
        return r