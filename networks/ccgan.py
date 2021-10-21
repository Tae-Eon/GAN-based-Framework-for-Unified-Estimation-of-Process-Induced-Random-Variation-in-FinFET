import torch
import torch.nn as nn
import numpy as np
#import torch.nn.functional as F #

#########################################################
# genearator
bias_g = False
class ccgen(nn.Module):
    def __init__(self, d_noise_num_of_input, gan_hidden_dim, num_of_output):
        super(ccgen, self).__init__()
        self.main_g = nn.Sequential(
                nn.Linear(d_noise_num_of_input, gan_hidden_dim, bias=bias_g),
                nn.BatchNorm1d(gan_hidden_dim),
                nn.ReLU(True),

                nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_g),
                nn.BatchNorm1d(gan_hidden_dim),
                nn.ReLU(True),

                nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_g),
                nn.BatchNorm1d(gan_hidden_dim),
                nn.ReLU(True),

                nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_g),
                nn.BatchNorm1d(gan_hidden_dim),
                nn.ReLU(True),

                nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_g),
                nn.BatchNorm1d(gan_hidden_dim),
                nn.ReLU(True),

                nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_g),
                nn.BatchNorm1d(gan_hidden_dim),
                nn.ReLU(True),

                nn.Linear(gan_hidden_dim, num_of_output, bias=bias_g),
            )

    def forward(self, noise, x):
        gen_input = torch.cat((noise, x), axis=1)
        r = self.main_g(gen_input)
        return r

#########################################################
# discriminator
bias_d=False
class ccdis(nn.Module):
    def __init__(self, num_of_output, gan_hidden_dim):
        super(ccdis, self).__init__()

        self.main_d = nn.Sequential(
            nn.Linear(num_of_output, gan_hidden_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(gan_hidden_dim, gan_hidden_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(gan_hidden_dim, 1, bias=bias_d),
            nn.Sigmoid()
        )


    def forward(self, y, x):
        dis_input = torch.cat((y, x), axis=1)

        r = self.main_d(dis_input)
        return r

