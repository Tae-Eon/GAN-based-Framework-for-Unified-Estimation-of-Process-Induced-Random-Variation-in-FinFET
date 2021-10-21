from __future__ import print_function

import networks
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

class GanTrainer(trainer.gan_GenericTrainer):
    def __init__(self, noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping):
        super().__init__(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping)
                        
    def train(self):
        
        p_real_list = []
        p_fake_list = []
        
        self.G.train()
        self.D.train()
        
        for i, data in enumerate(self.train_iterator):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            mini_batch_size = len(data_x)
            
            # GENERATOR
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
                        
            gen_y = self.G(z, data_x)
            
            c_fake = self.D(gen_y, data_x)
            
            g_loss = -1*c_fake.mean()
            
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()
            
            # DISCRIMINATOR
            
            
            # Loss for real data
            c_real = self.D(data_y, data_x)
            c_real_loss = -c_real.mean()
            
            # Loss for fake data
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
            
            gen_y = self.G(z, data_x)
            c_fake = self.D(gen_y, data_x)
            c_fake_loss = c_fake.mean()         
    
            c_loss = c_real_loss + c_fake_loss # maximize E[f(x)] - E[f(g(z))]
            
            self.optimizer_D.zero_grad()
            c_loss.backward()
            self.optimizer_D.step()
                        
            for p in self.D.parameters():
                p.data.clamp_(-1*self.clipping, self.clipping)
            
        self.prob['p_real_train'].append(c_real)
        self.prob['p_fake_train'].append(c_fake)
            
        for param_group in self.optimizer_D.param_groups:
            self.current_d_lr = param_group['lr']
        self.exp_gan_lr_scheduler.step()
        
        return c_real, c_fake
                    
    def evaluate(self):
        
        c_real, c_fake = 0., 0.
        batch_num = 0
        
        self.G.eval()
        self.D.eval()
        
        for i, data in enumerate(self.val_iterator):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
            
            with torch.autograd.no_grad():
                c_real += torch.sum(self.D(data_y, data_x)/mini_batch_size)
                
                gen_y = self.G(z, data_x)
                
                c_fake += torch.sum(self.D(gen_y, data_x)/mini_batch_size)
                
            batch_num += 1
            
        c_real /= batch_num
        c_fake /= batch_num
        
        self.prob['p_real_val'].append(c_real)
        self.prob['p_fake_val'].append(c_fake)
        
        return c_real, c_fake