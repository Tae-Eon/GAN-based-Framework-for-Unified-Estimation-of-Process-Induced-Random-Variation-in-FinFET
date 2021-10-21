import torch

import pandas as pd
import numpy as np

import os

def load_data_3(name, tr_num_in_cycle):
    
    data = np.load('./data_handler/'+name+'.npy', allow_pickle=True)

    X_all, Y_all, X_per_cycle, Y_per_cycle, Y_mean_cov = data[0], data[1], data[2], data[3], data[4]

    print("============ Train Data load =============")
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)
    print("Y mean cov shape : ", Y_mean_cov.shape)
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))

    original_num_in_cycle = int(len(X_all)/len(X_per_cycle))

    splited_X_all = np.array(np.split(X_all, len(X_per_cycle), axis=0))
    splited_Y_all = np.array(np.split(Y_all, len(Y_per_cycle), axis=0))
    reduced_splited_X_all = np.zeros((len(X_per_cycle), tr_num_in_cycle, X_all[0].shape[0]))
    reduced_splited_Y_all = np.zeros((len(Y_per_cycle), tr_num_in_cycle, Y_all[0].shape[0]))
    
    for i in range(len(splited_X_all)):
        reduced_splited_X_all[i] = splited_X_all[i][:tr_num_in_cycle]
        reduced_splited_Y_all[i] = splited_Y_all[i][:tr_num_in_cycle]

    reduced_X_all = np.concatenate((reduced_splited_X_all), axis=0)
    reduced_Y_all = np.concatenate((reduced_splited_Y_all), axis=0)
    reduced_X_per_cycle = np.mean(reduced_splited_X_all, axis=1)
    reduced_Y_per_cycle = np.mean(reduced_splited_Y_all, axis=1)

    return reduced_X_all, reduced_Y_all, reduced_X_per_cycle, reduced_Y_per_cycle, Y_mean_cov

def load_data_2(name, tr_num_in_cycle):
    
    data = np.load('./data_handler/'+name+'.npy', allow_pickle=True)

    X_all, Y_all, X_per_cycle, Y_per_cycle = data[0], data[1], data[2], data[3]

    print("============ Train Data load =============")
    print("X data shape: ", X_all.shape, "X per cycle data shape:", X_per_cycle.shape)
    print("Y data shape: ", Y_all.shape, "Y per cycle data shape:", Y_per_cycle.shape)  
    print("any nan in X?: ", np.argwhere(np.isnan(X_all)))
    print("any nan in Y?: ", np.argwhere(np.isnan(Y_all)))

    original_num_in_cycle = int(len(X_all)/len(X_per_cycle))

    splited_X_all = np.array(np.split(X_all, len(X_per_cycle), axis=0))
    splited_Y_all = np.array(np.split(Y_all, len(Y_per_cycle), axis=0))
    reduced_splited_X_all = np.zeros((len(X_per_cycle), tr_num_in_cycle, X_all[0].shape[0]))
    reduced_splited_Y_all = np.zeros((len(Y_per_cycle), tr_num_in_cycle, Y_all[0].shape[0]))
    
    for i in range(len(splited_X_all)):
        reduced_splited_X_all[i] = splited_X_all[i][:tr_num_in_cycle]
        reduced_splited_Y_all[i] = splited_Y_all[i][:tr_num_in_cycle]

    reduced_X_all = np.concatenate((reduced_splited_X_all), axis=0)
    reduced_Y_all = np.concatenate((reduced_splited_Y_all), axis=0)
    reduced_X_per_cycle = np.mean(reduced_splited_X_all, axis=1)
    reduced_Y_per_cycle = np.mean(reduced_splited_Y_all, axis=1)

    return reduced_X_all, reduced_Y_all, reduced_X_per_cycle, reduced_Y_per_cycle
        
def split_data(x, y, num_train, num_val):
    
    if len(x) == len(y):
        print("Same number of x data and y data")
        len_total = len(x)
    else:
        print("Different number of x data and y data")

    print(num_train, num_val)
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train+num_val], y[num_train:num_train+num_val]
    x_test, y_test = x[num_train+num_val:], y[num_train+num_val:]
    
    print("============= Train val Data split ==============")
    print("train X: {} train Y: {}".format(x_train.shape, y_train.shape))
    print("val X: {} val Y: {}".format(x_val.shape, y_val.shape))
    print("test X: {} test Y: {}".format(x_test.shape, y_test.shape))
    
    return x_train, y_train, x_val, y_val, x_test, y_test

class Dataset():   
    def __init__(self, name):
        
        self.train_X = None
        self.val_X = None
        self.test_X = None        
        
        self.train_Y = None
        self.val_Y = None
        self.test_Y = None
        
        self.train_X_per_cycle = None
        self.val_X_per_cycle = None
        self.test_X_per_cycle = None
              
        self.train_Y_per_cycle = None
        self.val_Y_per_cycle = None
        self.test_Y_per_cycle = None
        
        self.train_X_mean = None
        self.train_X_std = None
        
        self.train_Y_mean = None
        self.train_Y_std = None
        

class SEMI_gan_data(Dataset):
    def __init__(self, name, num_in_cycle, num_of_cycle, num_train, num_val):
        super().__init__(name)
        
        # STEP 1: load data
        
        X_all, Y_all, X_per_cycle, Y_per_cycle = load_data_2(name, num_in_cycle)
        
        self.train_X_mean = np.mean(X_all, axis=0, dtype=np.float32)
        self.train_X_std = np.std(X_all, axis=0, dtype=np.float32)
        
        self.train_Y_mean = np.mean(Y_all, axis=0, dtype=np.float32)
        self.train_Y_std = np.std(Y_all, axis=0, dtype=np.float32)
        
        print("X mean :", self.train_X_mean)
        print("X std :", self.train_X_std)
        print("Y mean :", self.train_Y_mean)
        print("Y std :", self.train_Y_std)
        
        # STEP 2: Split data

        self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle)
        self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val)
        
        
class SEMI_gaussian_data(Dataset):
    def __init__(self, name, num_in_cycle, num_of_cycle, num_train, num_val):
        super().__init__(name)
        
        # STEP 1: load data
        
        X_all, Y_all, X_per_cycle, Y_per_cycle, Y_mean_cov = load_data_3(name, num_in_cycle)
        
        self.train_X_mean = np.mean(X_all, axis=0, dtype=np.float32)
        self.train_X_std = np.std(X_all, axis=0, dtype=np.float32)
        
        self.train_Y_mean = np.mean(Y_all, axis=0, dtype=np.float32)
        self.train_Y_std = np.std(Y_all, axis=0, dtype=np.float32)
        
        print("X mean :", self.train_X_mean)
        print("X std :", self.train_X_std)
        print("Y mean :", self.train_Y_mean)
        print("Y std :", self.train_Y_std)
        
        # STEP 2: Split data

        self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = split_data(X_all, Y_all, num_train*num_in_cycle, num_val*num_in_cycle)
        self.train_X_per_cycle, self.train_Y_per_cycle, self.val_X_per_cycle, self.val_Y_per_cycle, self.test_X_per_cycle, self.test_Y_per_cycle = split_data(X_per_cycle, Y_per_cycle, num_train, num_val)
        self.train_X_per_cycle, self.train_Y_mean_cov, self.val_X_per_cycle, self.val_Y_mean_cov, self.test_X_per_cycle, self.test_Y_mean_cov  = split_data(X_per_cycle, Y_mean_cov, num_train, num_val)
        

class SEMI_sample_data(Dataset):
    def __init__(self, name):
        super().__init__(name)
        
        data = np.load('./data_handler/'+name+'.npy', allow_pickle=True)
    
        X_all, Y_all, X_per_cycle, Y_per_cycle = data[0], data[1], data[2], data[3]
        
        print("============ Test Data load =============")
        print("test X data shape: ", X_all.shape, "test X per cycle data shape:", X_per_cycle.shape)
        print("test Y data shape: ", Y_all.shape, "test Y per cycle data shape:", Y_per_cycle.shape)  
        print("any nan in test X?: ", np.argwhere(np.isnan(X_all)))
        print("any nan in test Y?: ", np.argwhere(np.isnan(Y_all)))
                
        self.test_X = X_all
        self.test_Y = Y_all
        self.test_X_per_cycle = X_per_cycle
        self.test_Y_per_cycle = Y_per_cycle