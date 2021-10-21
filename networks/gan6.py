import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class gen6(nn.Module):
    def __init__(self, d_noise_num_of_input, gan_hidden_dim, num_of_output):
        super(gen6, self).__init__()
        self.fc1 = nn.Linear(d_noise_num_of_input, gan_hidden_dim)
        self.fc2 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc3 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc4 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc5 = nn.Linear(gan_hidden_dim, num_of_output)
        
    def forward(self, noise, x):
        gen_input = torch.cat((noise, x), axis=1)
        r = F.relu(self.fc1(gen_input))
        r = F.relu(self.fc2(r))
        r = F.relu(self.fc3(r))
        r = F.relu(self.fc4(r))
        r = self.fc5(r)
        
        return r
        
class dis6(nn.Module):
    def __init__(self, num_of_output, gan_hidden_dim):
        super(dis6, self).__init__()
        self.fc1 = nn.Linear(num_of_output, gan_hidden_dim)
        self.fc2 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc3 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc4 = nn.Linear(gan_hidden_dim, gan_hidden_dim)
        self.fc5 = nn.Linear(gan_hidden_dim, 1)
        
    def forward(self, y, x):
        dis_input = torch.cat((y, x), axis=1)
        r = F.relu(self.fc1(dis_input))
        r = F.relu(self.fc2(r))
        r = F.relu(self.fc3(r))
        r = F.relu(self.fc4(r))
        r = torch.sigmoid(self.fc5(r))
        
        return r