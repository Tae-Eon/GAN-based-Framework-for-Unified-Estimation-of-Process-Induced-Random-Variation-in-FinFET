import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_of_input, num_of_output):
        super(Net, self).__init__()
        print(num_of_input, num_of_output)
        self.fc1 = nn.Linear(num_of_input, num_of_output)
        
    def forward(self, x):
        x=self.fc1(x)
        #x=F.sigmoid(x)

        return x    