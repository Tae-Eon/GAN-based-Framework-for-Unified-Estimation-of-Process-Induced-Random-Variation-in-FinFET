from __future__ import print_function

import networks
import trainer

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

from tqdm import tqdm
import copy

class MeanTrainer(trainer.mean_GenericTrainer):
    def __init__(self, train_iterator, val_iterator, mean_model, optimizer, exp_lr_scheduler):
        super().__init__(train_iterator, val_iterator, mean_model, optimizer, exp_lr_scheduler)
            
        self.best_loss = np.inf
        self.best_model = None
            
    def train(self):
        
        train_loss_list = []
        train_loss = 0
        train_num = 0
        self.model.train()
        
        for i, data in enumerate(self.train_iterator):
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            output = self.model(data_x)
            loss = F.mse_loss(output, data_y, reduction='mean')
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss
            train_num += mini_batch_size
            
        train_loss /= train_num
        self.loss['train_loss'].append(train_loss)
            
        for param_group in self.optimizer.param_groups:
            self.current_lr = param_group['lr']
        self.exp_lr_scheduler.step()
        
        return train_loss
            
    def evaluate(self):

        val_loss_list = []
        val_loss = 0.0
        val_num = 0
        self.model.eval()
        
        true_arr = []
        pred_arr = []
        
        with torch.no_grad():
            for i, data in enumerate(self.val_iterator):
                data_x, data_y = data
                data_x, data_y = data_x.cuda(), data_y.cuda()
                
                mini_batch_size = len(data_x)
                
                output = self.model(data_x)
                loss = F.mse_loss(output, data_y, reduction='mean')
                
                val_loss += loss
                val_num += len(data_y)
                
                true_arr += (data_y.data.cpu().numpy()).tolist()
                pred_arr += (output.data.cpu().numpy()).tolist()
                
                #print(true_arr, pred_arr)
            val_loss /= val_num
            self.loss['val_loss'].append(val_loss)
            

       
#         val_r2 = r2_score(true_arr, pred_arr)
        
        if val_loss < self.best_loss:
            self.best_model = copy.deepcopy(self.model)
            self.best_loss = val_loss
            self.best_mean = pred_arr
            
                
        return val_loss     