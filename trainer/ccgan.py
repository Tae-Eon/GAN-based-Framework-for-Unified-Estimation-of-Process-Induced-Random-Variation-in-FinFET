#RDF
from __future__ import print_function

import networks
import trainer
import utils

import numpy as np
import torch
import torch.nn.functional as F
import copy

from tqdm import tqdm

class GanTrainer(trainer.gan_GenericTrainer):
    def __init__(self, noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping, kernel_sigma, kappa, threshold_type, one_hot, fix_generator, fix_discriminator): 
        super().__init__(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping, kernel_sigma, kappa, threshold_type, one_hot, fix_generator, fix_discriminator) 
        self.clipping = None
        
    def train(self):
        p_real_list = []
        p_fake_list = []
        
        if self.fix_generator == True:
            self.G.eval()
            for param in self.G.parameters():
                param.requires_grad = False
        else:
            self.G.train()
            for param in self.G.parameters():
                param.requires_grad = True
        
        if self.fix_discriminator == True:
            self.D.eval()
            for param in self.D.parameters():
                param.requires_grad = False
        else:
            self.D.train()
            for param in self.D.parameters():
                param.requires_grad = True
        
        train_labels = torch.from_numpy(self.train_iterator.dataset.data_x).type(torch.float).cuda() ## LER or RDF+onehot input 
        train_samples = torch.from_numpy(self.train_iterator.dataset.data_y).type(torch.float).cuda() ## random variation output
            
            
            
        if self.one_hot == 0: # LER, LRW
            num_of_output = train_labels.shape[1]
            max_x = torch.max(train_labels, dim=0)[0]
            min_x = torch.min(train_labels, dim=0)[0]
            mean_x = torch.mean(train_labels, dim=0)[0]
            
        else : # RDF
            train_labels_sub = train_labels[:,:-self.one_hot]
            train_labels_dummpy = train_labels[:,-self.one_hot:]
            num_of_output = train_labels_sub.shape[1]
            max_x = torch.max(train_labels_sub, dim=0)[0]
            min_x = torch.min(train_labels_sub, dim=0)[0]
            mean_x = torch.mean(train_labels_sub, dim=0)[0]
            
        
        
        
        for i, data in enumerate(self.train_iterator):
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            batch_labels_dummpy = data_x[:,-self.one_hot:]#.cuda()
            
            mini_batch_size = len(data_x)
            
            
            
            ############### ccgan data gathering
            
            batch_epsilons = torch.from_numpy(np.random.normal(0, self.kernel_sigma, mini_batch_size)).type(torch.float).cuda() ##iteration 마다 랜덤한 margin 선택
            if self.one_hot == 0: # LER, LRW
                batch_target_labels = data_x + batch_epsilons.view(-1,1) ## kernel_sigma가 0이면 그대로
                batch_real_indx = torch.zeros(mini_batch_size, dtype=int)
            else : # RDF
                batch_target_labels_sub = data_x[:,:-self.one_hot] + batch_epsilons.view(-1,1)
                batch_target_labels = torch.cat((batch_target_labels_sub, batch_labels_dummpy), dim=1).cuda()
                batch_real_indx = torch.zeros(mini_batch_size, dtype=int)
                
            
            #################################################
            batch_real_samples = copy.deepcopy(data_y).cuda()
            for j in range(mini_batch_size):
#                 print('')
#                 print('before', batch_real_samples[j])
                if self.threshold_type == "hard":
                    if self.one_hot == 0 : # LER, LRW
                        distance = torch.sum(torch.abs(train_labels-batch_target_labels[j]), dim=1)
                        indx_real_in_vicinity = torch.where( (distance <= self.kappa * num_of_output) )[0]
                        if len(indx_real_in_vicinity)!=0:
                            selected_index = torch.argmin(distance + 0.000001*torch.randn(len(distance)).cuda())
                            batch_real_samples[j] = train_samples[selected_index]
                                                
                    else : # RDF
                        distance = torch.sum(torch.abs(train_labels_sub-batch_target_labels_sub[j]), dim=1)
                        indx_real_in_vicinity = torch.where( (distance>0.00001) * (distance <= self.kappa * num_of_output) )[0]
                        if len(indx_real_in_vicinity)!=0:
                            selected_index = torch.argmin(distance + 0.000001*torch.randn(len(distance)).cuda())
                            batch_real_samples[j] = train_samples[selected_index]
            
                else:
                    raise "not implemented"
                
#                 print('after ', batch_real_samples[j])
#                 print('')
   
            #################################################
        


            ## generate the fake image batch
            z = torch.randn(mini_batch_size, self.noise_d, dtype=torch.float).cuda()
            batch_fake_samples = self.G(z, data_x)

            # forward pass
            p_real_D = self.D(batch_real_samples, batch_target_labels)
            p_fake_D = self.D(batch_fake_samples, batch_target_labels)

            d_loss = - torch.mean(torch.log(p_real_D.view(-1)+1e-20)) - torch.mean(torch.log(1 - p_fake_D.view(-1)+1e-20))

            self.optimizer_D.zero_grad()
            if self.fix_discriminator == False:
                d_loss.backward()
            self.optimizer_D.step()
            
            
            ############### GENERATOR
            batch_epsilons = torch.from_numpy(np.random.normal(0, self.kernel_sigma, mini_batch_size)).type(torch.float).cuda() ##iteration 마다 랜덤한 margin 선택
            
            if self.one_hot == 0 :
                batch_target_labels = data_x + (mean_x*batch_epsilons.view(-1,1)) ## (normalized LER)
            else :      
                batch_target_labels_sub = data_x[:,:-self.one_hot] + (mean_x*batch_epsilons.view(-1,1))
                batch_target_labels = torch.cat((batch_target_labels_sub, batch_labels_dummpy), dim=1).cuda()
            
            z = utils.sample_z(mini_batch_size, self.noise_d).cuda()
            batch_fake_samples = self.G(z, batch_target_labels) ## unnormalized LER
            
            
            # loss
            p_fake = self.D(batch_fake_samples, batch_target_labels)
            g_loss = - torch.mean(torch.log(p_fake+1e-20))

            
            self.optimizer_G.zero_grad()
            if self.fix_generator == False:
                g_loss.backward()
            self.optimizer_G.step()
            
            
        self.prob['p_real_train'].append(p_real_D)
        self.prob['p_fake_train'].append(p_fake_D)
            
        for param_group in self.optimizer_D.param_groups:
            self.current_d_lr = param_group['lr']
        self.exp_gan_lr_scheduler.step()
        
        return p_real_D, p_fake_D
                    
    def evaluate(self, mode):

            if mode == 'train':
                iterator = self.train_iterator
            elif mode == 'test':
                iterator = self.eval_iterator

            p_real, p_fake = 0., 0.
            batch_num = 0

            self.G.eval()
            self.D.eval()

            for i, data in enumerate(iterator):

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