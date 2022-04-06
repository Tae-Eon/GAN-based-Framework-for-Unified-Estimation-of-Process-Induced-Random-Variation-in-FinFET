from __future__ import print_function

import networks
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

class GanTrainer(trainer.gan_GenericTrainer):
    def __init__(self, noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d):
        super().__init__(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d)
                
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
            p_fake = self.D(gen_y, data_x)
            g_loss = -1*torch.log(p_fake).mean()
            
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()
            
            # DISCRIMINATOR
            
            
            # Loss for real data
            p_real = self.D(data_y, data_x)
            d_real_loss = -1*torch.log(p_real).mean()
            
            # Loss for fake data
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
            
            gen_y = self.G(z, data_x)
            p_fake = self.D(gen_y, data_x)
            d_fake_loss = -1*torch.log(1.-p_fake).mean()            
    
            d_loss = (d_real_loss + d_fake_loss)/2
            
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()
            
        self.prob['p_real_train'].append(p_real)
        self.prob['p_fake_train'].append(p_fake)
            
        for param_group in self.optimizer_D.param_groups:
            self.current_d_lr = param_group['lr']
        self.exp_gan_lr_scheduler.step()
        
        return p_real, p_fake
                    
    def evaluate(self):
        
        p_real, p_fake = 0., 0.
        batch_num = 0
        
        self.G.eval()
        self.D.eval()
        
        for i, data in enumerate(self.val_iterator):
            
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            z = utils.sample_z(mini_batch_size, self.noise_d)
            
            with torch.autograd.no_grad():
                p_real += torch.sum(self.D(data_y, data_x)/mini_batch_size)
                
                gen_y = self.G(z, data_x)
                
                p_fake += torch.sum(self.D(gen_y, data_x)/mini_batch_size)
                
            batch_num += 1
            
        p_real /= batch_num
        p_fake /= batch_num
        
        self.prob['p_real_val'].append(p_real)
        self.prob['p_fake_val'].append(p_fake)
        
        return p_real, p_fake