import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hidden_dim, layer, num_of_input, num_of_output):
        super(Net, self).__init__()
        
        self.layer = layer
        
        print(num_of_input, layer, hidden_dim, num_of_output)
        self.fc1 = nn.Linear(num_of_input, hidden_dim)
        self.hidden = []
        for k in range(self.layer):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.out = nn.Linear(hidden_dim, num_of_output)
       
        
    def forward(self, x):
        x=torch.tanh(self.fc1(x))
        for layer in self.hidden:
            x=torch.tanh(layer(x))
        x=self.out(x)

        return x    