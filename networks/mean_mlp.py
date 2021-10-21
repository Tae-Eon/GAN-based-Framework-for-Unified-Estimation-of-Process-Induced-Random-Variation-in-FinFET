import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, mean_hidden_dim, num_of_input, num_of_output):
        super(Net, self).__init__()
        print(num_of_input, mean_hidden_dim)
        self.fc1 = nn.Linear(num_of_input, mean_hidden_dim)
        self.fc2 = nn.Linear(mean_hidden_dim, mean_hidden_dim)
        self.fc3 = nn.Linear(mean_hidden_dim, mean_hidden_dim)
        self.fc4 = nn.Linear(mean_hidden_dim, num_of_output)
        
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        #x=F.dropout(x, training=self.training)
        x=self.fc4(x)
        return x    