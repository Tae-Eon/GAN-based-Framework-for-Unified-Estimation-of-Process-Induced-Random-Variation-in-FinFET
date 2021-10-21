import torch
from torch.utils.data import DataLoader, TensorDataset

from torchvision import transforms
from scipy import linalg

from scipy import stats
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import cv2

import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.metrics import r2_score
import os
import ot
        
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

def FID_score_each_X(generated_samples, real_samples, num_in_gen, num_in_cycle):
    num_of_cycle = int(len(real_samples)/num_in_cycle)
    FID_score_list = []
    for i in range(num_of_cycle) :
        generated_samples_cycle = generated_samples[i*num_in_gen : (i+1)*num_in_gen]
        real_samples_cycle = real_samples[i*num_in_cycle : (i+1)*num_in_cycle]
        FID = FID_score(generated_samples_cycle, real_samples_cycle)
        FID_score_list.append(FID)
    return np.mean(np.array(FID_score_list)), FID_score_list

# def EMD_from_2d_kde_integral(generated_samples, real_samples, index_x, index_y, num_coordinate): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
#     y_min = min(min(real_samples[:,index_y]), min(generated_samples[:,index_y]))
#     y_max = max(max(real_samples[:,index_y]), max(generated_samples[:,index_y]))

#     position_cartesian = np.array(np.meshgrid(np.linspace(x_min, x_max, num_coordinate), np.linspace(y_min, y_max, num_coordinate))).T.reshape(-1,2)

#     real_samples_2d = real_samples[:,[index_x, index_y]]
#     generated_samples_2d = generated_samples[:,[index_x, index_y]]

#     kde_real = stats.gaussian_kde(real_samples_2d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_2d.T)

#     # Estimate distribution
#     # density_real = kde_real(position_cartesian.T) #2by100 입력 -> 1by100 출력
#     # density_gen = kde_gen(position_cartesian.T)



#     # Estimate probability
#     interval_x = (x_max - x_min)/num_coordinate
#     interval_y = (y_max - y_min)/num_coordinate
#     interval = np.array([interval_x, interval_y])

#     integral_real = np.zeros((num_coordinate**2, 1))
#     integral_gen = np.zeros((num_coordinate**2, 1))
#     for i in range(num_coordinate**2):
#         integral_real[i] = kde_real.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
#         integral_gen[i] = kde_gen.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

#     #clipping
#     integral_real = np.maximum(integral_real,0)
#     integral_gen = np.maximum(integral_gen,0)


#     real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')


#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

#     return EMD_score

# def EMD_from_1d_kde_integral(generated_samples, real_samples, index_x, num_coordinate): # 적분 weighting 버전

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))


#     position_cartesian = np.linspace(x_min, x_max, num_coordinate).reshape(-1,1)

#     real_samples_1d = real_samples[:,[index_x]]
#     generated_samples_1d = generated_samples[:,[index_x]]

#     kde_real = stats.gaussian_kde(real_samples_1d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_1d.T)

#     # Estimate distribution
#     #density_real = kde_real(position_cartesian.T)
#     #density_gen = kde_gen(position_cartesian.T)


#     # Estimate probability
#     interval = (x_max - x_min)/num_coordinate
#     integral_real = np.zeros((num_coordinate,1))
#     integral_gen = np.zeros((num_coordinate,1))
#     for i in range(num_coordinate):
#         integral_real[i] = kde_real.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
#         integral_gen[i] = kde_gen.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

#     #real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     #gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')

#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

#     return EMD_score

# def EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

#     dim = normalized_real_samples.shape[1]
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(dim):
#         for j in range(dim):
#             if i == j :
#                 EMD_1d_score_list.append(EMD_from_1d_kde_integral(normalized_generated_samples, normalized_real_samples, i, num_coordinate))
#             elif i < j : 
#                 EMD_2d_score_list.append(EMD_from_2d_kde_integral(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate))

#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_all_pair_each_X_integral(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
#     dim = real_samples.shape[1]
    
#     #EMD_score_list = []
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(num_of_cycle):
#         generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
#         real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
#         xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
#         xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
#         normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
#         normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
#         EMD_1d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate)[0])
#         EMD_2d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate)[1])
        
#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_from_2d_kde_pdf(generated_samples, real_samples, index_x, index_y, num_coordinate): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
#     y_min = min(min(real_samples[:,index_y]), min(generated_samples[:,index_y]))
#     y_max = max(max(real_samples[:,index_y]), max(generated_samples[:,index_y]))

#     position_cartesian = np.array(np.meshgrid(np.linspace(x_min, x_max, num_coordinate), np.linspace(y_min, y_max, num_coordinate))).T.reshape(-1,2)
    
#     real_samples_2d = real_samples[:,[index_x, index_y]]
#     generated_samples_2d = generated_samples[:,[index_x, index_y]]
    
#     kde_real = stats.gaussian_kde(real_samples_2d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_2d.T)
    
#     # Estimate distribution
#     density_real = kde_real(position_cartesian.T)
#     density_gen = kde_gen(position_cartesian.T)
    
#     real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')

#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    
#     return EMD_score


# def EMD_from_1d_kde_pdf(generated_samples, real_samples, index_x, num_coordinate): #대각 성분에 대해서는 1d kde로..

#     x_min = min(min(real_samples[:,index_x]), min(generated_samples[:,index_x]))
#     x_max = max(max(real_samples[:,index_x]), max(generated_samples[:,index_x]))
    

#     position_cartesian = np.linspace(x_min, x_max, num_coordinate).reshape(-1,1)
    
#     real_samples_1d = real_samples[:,[index_x]]
#     generated_samples_1d = generated_samples[:,[index_x]]
    
#     kde_real = stats.gaussian_kde(real_samples_1d.T)
#     kde_gen = stats.gaussian_kde(generated_samples_1d.T)
    
#     # Estimate distribution
#     density_real = kde_real(position_cartesian.T)
#     density_gen = kde_gen(position_cartesian.T)
    
#     real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
#     gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')

#     # Compare
#     #KL_div = KL(density_real, density_gen)
#     #print('KL_div', KL_div)

#     EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    
#     return EMD_score

# def EMD_1d_2d_list_pdf(normalized_generated_samples, normalized_real_samples, num_coordinate): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

#     dim = normalized_real_samples.shape[1]
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(dim):
#         for j in range(dim):
#             if i == j :
#                 EMD_1d_score_list.append(EMD_from_1d_kde_pdf(normalized_generated_samples, normalized_real_samples, i, num_coordinate))
#             elif i < j : 
#                 EMD_2d_score_list.append(EMD_from_2d_kde_pdf(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate))

#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_all_pair_each_X_pdf(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
#     dim = real_samples.shape[1]
    
#     #EMD_score_list = []
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []

#     for i in range(num_of_cycle):
#         generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
#         real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
#         xy_mean = np.mean(real_samples_cycle, axis=0, dtype=np.float32)
#         xy_std = np.std(real_samples_cycle, axis=0, dtype=np.float32)
        
#         normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
#         normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std
        
#         EMD_1d_score_list.append(EMD_1d_2d_list_pdf(normalized_generated_samples, normalized_real_samples, num_coordinate)[0])
#         EMD_2d_score_list.append(EMD_1d_2d_list_pdf(normalized_generated_samples, normalized_real_samples, num_coordinate)[1])
        
#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


def EMD_from_2d_kde_integral(normalized_generated_samples, normalized_real_samples, index_x, index_y, num_coordinate, normalized_x_min, normalized_x_max, normalized_y_min, normalized_y_max): #각 joint pair 별로 2d sample을 kde로 fittining시키고 얻은 pdf값을 weight로 삼아 histogram 만들어서 EMD

    position_cartesian = np.array(np.meshgrid(np.linspace(normalized_x_min, normalized_x_max, num_coordinate), np.linspace(normalized_y_min, normalized_y_max, num_coordinate))).T.reshape(-1,2)

    normalized_real_samples_2d = normalized_real_samples[:,[index_x, index_y]]
    normalized_generated_samples_2d = normalized_generated_samples[:,[index_x, index_y]]

    kde_real = stats.gaussian_kde(normalized_real_samples_2d.T, bw_method='silverman') # silverman
    kde_gen = stats.gaussian_kde(normalized_generated_samples_2d.T, bw_method='silverman')

    # Estimate distribution
    # density_real = kde_real(position_cartesian.T) #2by100 입력 -> 1by100 출력
    # density_gen = kde_gen(position_cartesian.T)



    # Estimate probability
    interval_x = (normalized_x_max - normalized_x_min)/(num_coordinate-1)
    interval_y = (normalized_y_max - normalized_y_min)/(num_coordinate-1)
    interval = np.array([interval_x, interval_y])

    integral_real = np.zeros((num_coordinate**2, 1))
    integral_gen = np.zeros((num_coordinate**2, 1))
    for i in range(num_coordinate**2):
        integral_real[i] = kde_real.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
        integral_gen[i] = kde_gen.integrate_box(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

    #clipping
    integral_real = np.maximum(integral_real,0)
    integral_gen = np.maximum(integral_gen,0)


    real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')


    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)

    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

    return EMD_score

def EMD_from_1d_kde_integral(normalized_generated_samples, normalized_real_samples, index_x, num_coordinate, normalized_x_min, normalized_x_max): # 적분 weighting 버전

    position_cartesian = np.linspace(normalized_x_min, normalized_x_max, num_coordinate).reshape(-1,1)

    normalized_real_samples_1d = normalized_real_samples[:,[index_x]]
    normalized_generated_samples_1d = normalized_generated_samples[:,[index_x]]

    kde_real = stats.gaussian_kde(normalized_real_samples_1d.T, bw_method='silverman')
    kde_gen = stats.gaussian_kde(normalized_generated_samples_1d.T, bw_method='silverman')

    # Estimate distribution
    #density_real = kde_real(position_cartesian.T)
    #density_gen = kde_gen(position_cartesian.T)


    # Estimate probability
    interval = (normalized_x_max - normalized_x_min)/(num_coordinate-1)
    integral_real = np.zeros((num_coordinate,1))
    integral_gen = np.zeros((num_coordinate,1))
    for i in range(num_coordinate):
        integral_real[i] = kde_real.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)
        integral_gen[i] = kde_gen.integrate_box_1d(position_cartesian[i]-interval/2, position_cartesian[i]+interval/2)

    #real_weight_position = np.concatenate((density_real.reshape(-1,1), position_cartesian),axis=1).astype('float32')
    #gen_weight_position = np.concatenate((density_gen.reshape(-1,1), position_cartesian),axis=1).astype('float32')
    real_weight_position = np.concatenate((integral_real, position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((integral_gen, position_cartesian),axis=1).astype('float32')

    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)

    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)

    return EMD_score




def EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate, noramlized_min_list, noramlized_max_list): #하나의 X에 대해, 모든 pair(36가지) EMD를 list로 뱉는 함수.

    dim = normalized_real_samples.shape[1]
    EMD_1d_score_list = []
    EMD_2d_score_list = []

    for i in range(dim):
        for j in range(dim):
            if i == j :
                EMD_1d_score_list.append(EMD_from_1d_kde_integral(normalized_generated_samples, normalized_real_samples, i, num_coordinate, noramlized_min_list[i], noramlized_max_list[i]))
            elif i < j : 
                EMD_2d_score_list.append(EMD_from_2d_kde_integral(normalized_generated_samples, normalized_real_samples, i, j, num_coordinate, noramlized_min_list[i], noramlized_max_list[i], noramlized_min_list[j], noramlized_max_list[j]))

    return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)


# def EMD_all_pair_each_X_integral(generated_samples, real_samples, num_coordinate, num_of_cycle, num_in_gen_list, num_in_real_list, min_list, max_list): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
#     dim = real_samples.shape[1]
    
#     #EMD_score_list = []
#     EMD_1d_score_list = []
#     EMD_2d_score_list = []
    
#     xy_mean = np.mean(real_samples, axis=0, dtype=np.float32)
#     xy_std = np.std(real_samples, axis=0, dtype=np.float32)
#     normalized_min_list = (min_list-xy_mean)/xy_std
#     normalized_max_list = (max_list-xy_mean)/xy_std
    
#     for i in range(num_of_cycle):
#         generated_samples_cycle = generated_samples[sum(num_in_gen_list[:i]):sum(num_in_gen_list[:i])+num_in_gen_list[i]]
#         real_samples_cycle = real_samples[sum(num_in_real_list[:i]):sum(num_in_real_list[:i])+num_in_real_list[i]]
        
#         normalized_generated_samples = (generated_samples_cycle-xy_mean)/xy_std
#         normalized_real_samples = (real_samples_cycle-xy_mean)/xy_std

#         EMD_1d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate, normalized_min_list, normalized_max_list)[0])
#         EMD_2d_score_list.append(EMD_1d_2d_list_integral(normalized_generated_samples, normalized_real_samples, num_coordinate, normalized_min_list, normalized_max_list)[1])
        
#     return np.array(EMD_1d_score_list), np.array(EMD_2d_score_list)

def new_EMD_all_pair_each_X_integral(generated_samples, real_samples, real_bin_num, num_of_cycle, min_list, max_list, train_mean, train_std, minmax, check): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
    dim = real_samples.shape[1]
    
    EMD_score_list = []
    sink_score_list = []
    
    xy_mean = train_mean
    xy_std = train_std
    
    for factor in range(num_of_cycle):
        if check == True:
            print()
            print("factor:", factor)
        
        # samples
        generated_samples_cycle = generated_samples[factor]
        real_samples_cycle = real_samples[factor]
                
        # min, max list
        if 'local' in minmax:
            min_list_cycle = min_list[factor]
            max_list_cycle = max_list[factor]
            if check == True:
                print(min_list_cycle.shape, max_list_cycle.shape)
        #     normalized_min_list_cycle = normalized_min_list[factor]
        #     normalized_max_list_cycle = normalized_max_list[factor]
        elif 'global' in minmax:
            min_list_cycle = min_list
            max_list_cycle = max_list
        #     normalized_min_list_cycle = normalized_min_list
        #     normalized_max_list_cycle = normalized_max_list
        
        EMD_1D, EMD_2D, sink_1D, sink_2D = factor_wise_EMD_global(generated_samples_cycle, real_samples_cycle, min_list_cycle, max_list_cycle, xy_mean, xy_std, check, real_bin_num=real_bin_num)

        EMD_score_cat = np.hstack((EMD_1D, EMD_2D))
        sink_score_cat = np.hstack((sink_1D, sink_2D))

        EMD_score_list.append(EMD_score_cat)
        sink_score_list.append(sink_score_cat)
    
    EMD_score_list = np.array(EMD_score_list)
    sink_score_list = np.array(sink_score_list)

    EMD_score = np.mean(EMD_score_list, axis=1)
    sink_score = np.mean(sink_score_list, axis=1)


    return np.array(EMD_score_list), np.array(sink_score_list)

def EMD_test_v4(gen_samples, real_samples, index_x, index_y, x_max, y_max, x_min, y_min, real_min, real_max, real_bin_len, check, real_bin_num):
    """
    input
    gen_samples : (250, 2)
    real_samples : (250, 2)
    index_x, index_y : integer
    x_max, x_min, y_max, y_min : sample 기준
    """
    if check == True:
        print('x')
        print('global real min', real_min[index_x], 'sample real min', x_min)
        print('global real max', real_max[index_x], 'sample real max', x_max)

        print('y')
        print('global real min', real_min[index_y], 'sample real min', y_min)
        print('global real max', real_max[index_y], 'sample real max', y_max) 
    
    start_coord_x = 0
    start_coord_y = 0
    
    # consider starting point
    # index x
    if x_min < real_min[index_x]:
        diff_x = real_min[index_x] - x_min
        add_bin_num_x = int(diff_x//real_bin_len[index_x])+1
        start_coord_x = real_min[index_x] - real_bin_len[index_x]*(add_bin_num_x)
        if check== True:
            print('x', 'starting point', real_min[index_x], '->', start_coord_x, '|', 'sample real min', x_min, 'real len', real_bin_len[index_x])
    
    elif x_min >= real_min[index_x]:
        add_bin_num_x = 0
        start_coord_x = real_min[index_x]
    
        
    # index y 
    if y_min < real_min[index_y]:
        diff_y = real_min[index_y] - y_min
        add_bin_num_y = int(diff_y//real_bin_len[index_y])+1
        start_coord_y = real_min[index_y] - real_bin_len[index_y]*(add_bin_num_y)
        if check== True:
            print('y','starting point', real_min[index_y], '->', start_coord_y, '|', 'sample real min', y_min,  'real len', real_bin_len[index_y])
    
    elif y_min >= real_min[index_y]:
        add_bin_num_y = 0
        start_coord_y = real_min[index_y]
    
    # consider total bin number
    
    x_bin_num = 0
    y_bin_num = 0
    if check == True: 
        print('x sample max', x_max, 'x real max', real_max[index_x])
        print('y sample max', y_max, 'y real max', real_max[index_y])
        
    if real_max[index_x] < x_max:
        
        x_bin_num = int((x_max - start_coord_x) // real_bin_len[index_x]) + 1
        
    elif real_max[index_x] > x_max:
        
        x_bin_num = real_bin_num + add_bin_num_x
        
    elif real_max[index_x] == x_max:
    
        x_bin_num = real_bin_num + add_bin_num_x

    if real_max[index_y] < y_max:
        
        y_bin_num = int((y_max - start_coord_y) // real_bin_len[index_y]) + 1
        
    elif real_max[index_y] > y_max:
        
        y_bin_num = real_bin_num + add_bin_num_y

    elif real_max[index_y] == y_max:
        
        y_bin_num = real_bin_num + add_bin_num_y
        
    x_axis = np.array([start_coord_x+real_bin_len[index_x]*i for i in range(x_bin_num+1)])
    y_axis = np.array([start_coord_y+real_bin_len[index_y]*i for i in range(y_bin_num+1)])   

    if check== True:
        print('x_axis:', x_axis)
        print('y_axis:', y_axis)
        print('x bin num', x_bin_num, 'y bin num', y_bin_num)

    integral_real = np.zeros(((x_bin_num+1)*(y_bin_num+1), 1))
    integral_gen = np.zeros(((x_bin_num+1)*(y_bin_num+1), 1))
    
    normalized_x, normalized_y = np.meshgrid(x_axis, y_axis)
    normalized_position_cartesian = np.array([normalized_x.flatten(), normalized_y.flatten()]).T
    
    kde_real = stats.gaussian_kde(real_samples.T, bw_method='silverman') # silverman
    kde_gen = stats.gaussian_kde(gen_samples.T, bw_method='silverman')

    # Estimate distribution
    # density_real = kde_real(position_cartesian.T) #2by100 입력 -> 1by100 출력
    # density_gen = kde_gen(position_cartesian.T)

    # Estimate probability
    interval = np.array([real_bin_len[index_x], real_bin_len[index_y]])
    
#     for i in range((EMD_x_bin_num+1)*(EMD_y_bin_num+1)):
    for i in range((x_bin_num+1)*(y_bin_num+1)):

        integral_real[i] = kde_real.integrate_box(normalized_position_cartesian[i]-interval/2, normalized_position_cartesian[i]+interval/2)
        integral_gen[i] = kde_gen.integrate_box(normalized_position_cartesian[i]-interval/2, normalized_position_cartesian[i]+interval/2)

    #clipping
    integral_real = np.maximum(integral_real,1e-5)
    integral_gen = np.maximum(integral_gen,1e-5)

#     print(integral_real.shape)
#     print(normalized_position_cartesian.shape)
    real_weight_position = np.concatenate((integral_real, normalized_position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((integral_gen, normalized_position_cartesian),axis=1).astype('float32')
    
    # M : ground distance matrix
    coordsSqr = np.sum(normalized_position_cartesian**2, 1)
    M = coordsSqr[:, None] + coordsSqr[None, :] - 2*normalized_position_cartesian.dot(normalized_position_cartesian.T)
    M[M < 0] = 0
    M = np.sqrt(M)
#     print(M)

    
    if check == True:
        plt.scatter(real_samples[:,0],real_samples[:,1],color='blue')
        plt.scatter(gen_samples[:,0], gen_samples[:,1],color='orange')
        plt.scatter(normalized_position_cartesian[:,0],normalized_position_cartesian[:,1],color='black')
        plt.show()
    
    wass = 0
#     wass = sinkhorn2(integral_real, integral_gen, M, 1.0, numItermax=1000)

#     print(real_weight_position)
#     print(gen_weight_position)
    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)
      
    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    if check == True:
        print('EMD_score', EMD_score)
        print('sinkhorn', wass)
    
    return EMD_score, wass

def EMD_test_v5(gen_samples, real_samples, index, sample_max, sample_min, real_min, real_max, real_bin_len, check, real_bin_num):
    """
    input
    gen_samples : (250, 1)
    real_samples : (250, 1)
    index: integer
    sample_max, sample_min : sample 기준
    real_max, real_min : real 기준
    real_bin_len : interval의 length
    real_bin_num : bin num 개수
    """
    if check == True:
        print('global real min', real_min[index], 'sample real min', sample_min)
        print('global real max', real_max[index], 'sample real max', sample_max)

    start_coord = 0
    
    # consider starting point
    # index x
    if sample_min < real_min[index]:
        diff = real_min[index] - sample_min
        add_bin_num = int(diff//real_bin_len[index])+1
        start_coord = real_min[index] - real_bin_len[index]*(add_bin_num)
        if check == True:
            print('starting point', real_min[index], '->', start_coord, '|', 'sample real min', sample_min, 'real len', real_bin_len[index])
    elif sample_min >= real_min[index]:
        add_bin_num = 0
        start_coord = real_min[index]
    
     # consider total bin number
    
    bin_num = 0

    if real_max[index] < sample_max:
        
        bin_num = int((sample_max - start_coord) // real_bin_len[index]) + 1
        
    elif real_max[index] > sample_max:
        
        bin_num = real_bin_num + add_bin_num
        
    elif real_max[index] == sample_max:
    
        bin_num = real_bin_num + add_bin_num

    axis = np.array([start_coord+real_bin_len[index]*i for i in range(bin_num+1)])

    if check==True:
        
        print('sample max', sample_max, 'real max', real_max[index])
        print('axis:', axis)
        print('bin num:', bin_num)
        
    integral_real = np.zeros((bin_num+1, 1))
    integral_gen = np.zeros((bin_num+1, 1))

    
    normalized_position_cartesian = np.array(axis).reshape(-1,1)
    
    kde_real = stats.gaussian_kde(real_samples.T, bw_method='silverman') # silverman
    kde_gen = stats.gaussian_kde(gen_samples.T, bw_method='silverman')

    # Estimate distribution
    # density_real = kde_real(position_cartesian.T) #2by100 입력 -> 1by100 출력
    # density_gen = kde_gen(position_cartesian.T)

    # Estimate probability
    interval = real_bin_len[index]
    
    for i in range(bin_num+1):
        integral_real[i] = kde_real.integrate_box(normalized_position_cartesian[i]-interval/2, normalized_position_cartesian[i]+interval/2)
        integral_gen[i] = kde_gen.integrate_box(normalized_position_cartesian[i]-interval/2, normalized_position_cartesian[i]+interval/2)

    #clipping
    integral_real = np.maximum(integral_real,1e-7)
    integral_gen = np.maximum(integral_gen,1e-7)

#     print(integral_real.shape)
#     print(normalized_position_cartesian.shape)
    real_weight_position = np.concatenate((integral_real, normalized_position_cartesian),axis=1).astype('float32')
    gen_weight_position = np.concatenate((integral_gen, normalized_position_cartesian),axis=1).astype('float32')
    
    # M : ground distance matrix
    coordsSqr = np.sum(normalized_position_cartesian**2, 1)
    M = coordsSqr[:, None] + coordsSqr[None, :] - 2*normalized_position_cartesian.dot(normalized_position_cartesian.T)
    M[M < 0] = 0
    M = np.sqrt(M)
#     print(M)
    
#     print(normalized_position_cartesian.flatten().tolist())
    if check == True: 
        plt.plot(normalized_position_cartesian.flatten(),kde_real(normalized_position_cartesian.T), color='blue')
        plt.plot(normalized_position_cartesian.flatten(),kde_gen(normalized_position_cartesian.T), color='orange')
#         plt.hist(integral_real.flatten(), bins=[normalized_position_cartesian.flatten().tolist()], color='blue')
#         plt.hist(integral_gen.flatten(), bins=[normalized_position_cartesian.flatten().tolist()], color='orange')
        plt.show()
     
   
    wass = 0
#     wass = sinkhorn2(integral_real, integral_gen, M, 1.0, numItermax=1000)

#     print(real_weight_position)
#     print(gen_weight_position)
    # Compare
    #KL_div = KL(density_real, density_gen)
    #print('KL_div', KL_div)
      
    EMD_score, _, flow = cv2.EMD(real_weight_position, gen_weight_position, cv2.DIST_L2)
    if check==True:
        print('EMD_score', EMD_score)
        print('sinkhorn', wass)
    
    return EMD_score, wass

def factor_wise_EMD_global(generated_samples_cycle, real_samples_cycle, min_list_cycle, max_list_cycle, train_Y_mean, train_Y_std, check, real_bin_num):
    
    # Normalize samples
    normalized_generated_samples_cycle = (generated_samples_cycle-train_Y_mean)/train_Y_std
    normalized_real_samples_cycle = (real_samples_cycle-train_Y_mean)/train_Y_std
    
#     print(normalized_real_samples_cycle)
    # Normalize min, max
    normalized_min_list = (min_list_cycle-train_Y_mean)/train_Y_std
    normalized_max_list = (max_list_cycle-train_Y_mean)/train_Y_std

    interval = normalized_max_list - normalized_min_list
          
    real_bin_len = interval/real_bin_num
    if check == True:
        print('global interval', interval)
        print('global bin length', real_bin_len)
        print()
    
    num_of_output = normalized_generated_samples_cycle.shape[1]
    
    # 1 D
    
    EMD_1D = []
    sink_1D = []
    
    for i in range(num_of_output):
        
        index = i
        
        normalized_real_samples_cycle_control = normalized_real_samples_cycle[:, index]
        normalized_generated_samples_cycle_control = normalized_generated_samples_cycle[:, index]
        
        normalized_sample_min = np.min(np.concatenate((normalized_real_samples_cycle_control, normalized_generated_samples_cycle_control), axis=0), axis=0)
        normalized_sample_max = np.max(np.concatenate((normalized_real_samples_cycle_control, normalized_generated_samples_cycle_control), axis=0), axis=0)
    
        EMD, sink = EMD_test_v5(normalized_generated_samples_cycle_control, normalized_real_samples_cycle_control, index, normalized_sample_max, normalized_sample_min, normalized_min_list, normalized_max_list, real_bin_len, check, real_bin_num=real_bin_num)
        EMD_1D.append(EMD)
        sink_1D.append(sink)
        
    
    # 2 D
    EMD_2D = []
    sink_2D = []
#     print(num_of_output)
    
    count = 0
    
    for i in range(num_of_output):
        for j in range(num_of_output):
            if j>i:
                count+=1
                
                index_x = i
                index_y = j
                
                normalized_real_samples_cycle_control = normalized_real_samples_cycle[:, [index_x, index_y]]
                normalized_generated_samples_cycle_control = normalized_generated_samples_cycle[:, [index_x, index_y]]
                
                normalized_sample_min = np.min(np.concatenate((normalized_real_samples_cycle_control, normalized_generated_samples_cycle_control), axis=0), axis=0)
                normalized_sample_max = np.max(np.concatenate((normalized_real_samples_cycle_control, normalized_generated_samples_cycle_control), axis=0), axis=0)
                
                EMD, sink = EMD_test_v4(normalized_generated_samples_cycle_control, normalized_real_samples_cycle_control, index_x, index_y, normalized_sample_max[0], normalized_sample_max[1], normalized_sample_min[0], normalized_sample_min[1], normalized_min_list, normalized_max_list, real_bin_len, check, real_bin_num=real_bin_num)
                EMD_2D.append(EMD)
                sink_2D.append(sink)
                
    return np.array(EMD_1D).flatten(), np.array(EMD_2D).flatten(), np.array(sink_1D).flatten(), np.array(sink_2D).flatten()

def sinkhorn2(a, b, M, reg, method='sinkhorn', numItermax=1000,
              stopThr=1e-9, verbose=False, log=False, **kwargs):
    """
    Solve the entropic regularization optimal transport problem and return the loss
    The function solves the following optimization problem:
    .. math::
        W = \min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_
    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    method : str
        method used for the solver either 'sinkhorn',  'sinkhorn_stabilized', see those function for specific parameters
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **Choosing a Sinkhorn solver**
    By default and when using a regularization parameter that is not too small
    the default sinkhorn solver should be enough. If you need to use a small
    regularization to get sharper OT matrices, you should use the
    :any:`ot.bregman.sinkhorn_stabilized` solver that will avoid numerical
    errors. This last solver can be very slow in practice and might not even
    converge to a reasonable OT matrix in a finite time. This is why
    :any:`ot.bregman.sinkhorn_epsilon_scaling` that relie on iterating the value
    of the regularization (and using warm start) sometimes leads to better
    solutions. Note that the greedy version of the sinkhorn
    :any:`ot.bregman.greenkhorn` can also lead to a speedup and the screening
    version of the sinkhorn :any:`ot.bregman.screenkhorn` aim a providing  a
    fast approximation of the Sinkhorn problem.
    Returns
    -------
    W : (n_hists) ndarray
        Optimal transportation loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    Examples
    --------
    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn2(a, b, M, 1)
    array([0.26894142])
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
       [21] Altschuler J., Weed J., Rigollet P. : Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration, Advances in Neural Information Processing Systems (NIPS) 31, 2017
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    ot.bregman.sinkhorn_knopp : Classic Sinkhorn [2]
    ot.bregman.greenkhorn : Greenkhorn [21]
    ot.bregman.sinkhorn_stabilized: Stabilized sinkhorn [9][10]
    """
    b = np.asarray(b, dtype=np.float64)
    if len(b.shape) < 2:
        b = b[:, None]

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                              stopThr=stopThr, verbose=verbose, log=log,
                              **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, M, reg, numItermax=numItermax,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)

def sinkhorn_knopp(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_
    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    Examples
    --------
    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((dim_a, n_hists)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # print(reg)

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    # print(np.min(K))
    tmp2 = np.empty(b.shape, dtype=M.dtype)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)
#         print('v', v)
#         print('u', Kp)

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            
#             print(np.any(KtransposeU == 0))
#             print(np.any(np.isnan(u)))
#             print(np.any(np.isnan(v)))
#             print(np.any(np.isinf(u)))
#             print(np.any(np.isinf(v)))
            
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                np.einsum('ik,ij,jk->jk', u, K, v, out=tmp2)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                np.einsum('i,ij,j->j', u, K, v, out=tmp2)
            err = np.linalg.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if n_hists:  # return only loss
        res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))
