import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F    
from scipy import linalg
import random
import os
from sklearn import metrics
    
    
def train_mean_std(args, x, y):
    
    
    data_type = x[0][args.num_of_input:].reshape(1, -1) ##3은 ler, rdf, wfv -> num_of_input 바꿔주기
    #print('data_type',data_type.shape)

    x_mean = np.mean(x[:,:args.num_of_input], axis=0, dtype=np.float64)
    x_std = np.std(x[:,:args.num_of_input], axis=0, dtype=np.float64)
    
    x_mean = x_mean.reshape(1, args.num_of_input)
    x_std = x_std.reshape(1, args.num_of_input)
    
#     x_mean = np.hstack((x_mean, data_type))
#     x_std = np.hstack((x_std, data_type))
        
    y_mean = np.mean(y, axis=0, dtype=np.float64)
    y_std = np.std(y, axis=0, dtype=np.float64)
    
    return x_mean, x_std, y_mean, y_std

def normalize_train(x, y, x_mean, x_std, y_mean, y_std):
    
    norm_x = (x - x_mean) / x_std
    norm_y = (y - y_meah) / y_std
    
    return norm_x, norm_y

# def normalize(x, y):
        
#     x_mean = np.mean(x, axis=0, dtype=np.float32)
#     x_std = np.std(x, axis=0, dtype=np.float32)
        
#     y_mean = np.mean(y, axis=0, dtype=np.float32)
#     y_std = np.std(1e+10*y, axis=0, dtype=np.float32)

#     norm_x = ( x - x_mean ) / (x_std)
#     norm_y = ( y - y_mean )*1e+10 / (y_std)
       
#     y_std = y_std / 1e+10
#     return norm_x, norm_y, x_mean, x_std, y_mean, y_std

def init_params(model):
    for p in model.parameters():
        if (p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)    
    
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
       
    
def sample_z(batch_size = 1, d_noise=100):    
    return torch.randn(batch_size, d_noise).cuda()



def FID_score(generated_samples, real_samples):
    # https://en.wikipedia.org/wiki/Sample_mean_and_covariance
    mu_g = np.mean(generated_samples, axis=0, keepdims=True).T
    mu_r = np.mean(real_samples, axis=0, keepdims=True).T
    cov_g = (generated_samples - np.ones((len(generated_samples),1)).dot(mu_g.T)).T.dot((generated_samples - np.ones((len(generated_samples),1)).dot(mu_g.T)))/(len(generated_samples)-1)
    cov_r = (real_samples - np.ones((len(real_samples),1)).dot(mu_r.T)).T.dot((real_samples - np.ones((len(real_samples),1)).dot(mu_r.T)))/(len(real_samples)-1)

    
    mean_diff = mu_g - mu_r
    cov_prod_sqrt = linalg.sqrtm(cov_g.dot(cov_r))
    
    #numerical instability of linalg.sqrtm
    #https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
    eps=1e-6
    if not np.isfinite(cov_prod_sqrt).all():
        offset = np.eye(cov_g.shape[0]) * eps
        cov_prod_sqrt = linalg.sqrtm((cov_g + offset).dot(cov_r + offset))

    if np.iscomplexobj(cov_prod_sqrt):
        if not np.allclose(np.diagonal(cov_prod_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_prod_sqrt.imag))
            raise ValueError('Imaginary component {}'.format(m))
        cov_prod_sqrt = cov_prod_sqrt.real
    
    
    FID_score = np.sum(mean_diff**2) + np.trace(cov_g + cov_r -2*cov_prod_sqrt)
    return FID_score

def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def makedirs(path): 
    try: 
        os.makedirs(path) 
    except OSError: 
        if not os.path.isdir(path): 
            raise


            
def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
            
def calculate_MMD(gen_samples_set, real_samples, mean_list, std_list, gamma_list):
    
    num_of_seed = gen_samples_set.shape[0]
    ###################### Calculate MMD ######################
    test_MMD_score_list_set = []
    for seed in range(num_of_seed):
        test_gen = gen_samples_set[seed]
        test_MMD_score_list = []
        for i in range(len(real_samples)):
            gamma=gamma_list[i]
            test_gen_tmp = (test_gen[i] - mean_list)/std_list
            real_samples_tmp = (real_samples[i] - mean_list)/std_list
            test_MMD_score_list.append(mmd_rbf(test_gen_tmp, real_samples_tmp, gamma=gamma))    
        test_MMD_score_list_set.append(test_MMD_score_list)
    ###################### Add 'MMD value' to file #####################
    MMDs = np.array(test_MMD_score_list_set)
    
    return MMDs