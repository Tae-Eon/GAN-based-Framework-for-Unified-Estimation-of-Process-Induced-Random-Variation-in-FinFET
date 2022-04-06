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
import pickle

from sklearn import metrics
        
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from shapely.geometry import Polygon 
from numpy import linalg as LA

def confidence_ellipse_2(x, y, check, ax=None, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    egvalue, egvector = LA.eig(cov)
    order = np.argsort(egvalue)[::-1]

    sorted_egvalue = egvalue[order]
    sorted_egvector = egvector[:,order]

    theta = np.degrees(np.arctan2(*sorted_egvector[:,0][::-1]))

    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)

    ellipse = Ellipse((x_mean, y_mean), width=2*n_std*np.sqrt(sorted_egvalue[0]), height=2*n_std*np.sqrt(sorted_egvalue[1]),
                      facecolor=facecolor, angle=theta, **kwargs)
    if check==True:
        return ax.add_patch(ellipse), ellipse
    else:
        return ellipse

def ellipsoid_2d_2(test_gen, test_real, test_emd_list, n_std, datatype, factor, model, check):

    if datatype == "LER":
        label = ['Ioff(Normalized)', 'Idsat(Normalized)', 'Idlin(Normalized)', 'Vtsat(Normalized)', 'Vtlin(Normalized)', 'SS(Normalized)']
    elif datatype == "RDFWFV":
        label = ['Ioff(Normalized)', 'Vtlin(Normalized)', 'Vtsat(Normalized)', 'Idlin(Normalized)', 'Idsat(Normalized)', 'SS(Normalized)']
    elif datatype == "LERRDFWFV":
        label = ['Ioff(Normalized)', 'Vtlin(Normalized)', 'Vtsat(Normalized)', 'Idlin(Normalized)', 'Idsat(Normalized)', 'SS(Normalized)']
        
    sample_num = test_gen.shape[0]
    num_of_output = test_real.shape[1]

    iou_list = []

    cnt = 6
    for i in range(num_of_output):
        for j in range(num_of_output):
            if i < j :

                min_x_axis = np.min(np.concatenate((test_gen[:,j], test_real[:,j])))
                max_x_axis = np.max(np.concatenate((test_gen[:,j], test_real[:,j])))

                min_y_axis = np.min(np.concatenate((test_gen[:,i], test_real[:,i])))
                max_y_axis = np.max(np.concatenate((test_gen[:,i], test_real[:,i])))

                ax_nstd = None

                if check == True:
                    fig, ax_nstd = plt.subplots(figsize=(6, 6))
                    ax_nstd.axvline(c='grey', lw=1)
                    ax_nstd.axhline(c='grey', lw=1)

                    ax_nstd.scatter(test_gen[:,j], test_gen[:,i], c='r')
                    ax_nstd.scatter(test_real[:,j], test_real[:,i], c='b')

                    _, ellipse1 = confidence_ellipse_2(test_gen[:,j], test_gen[:,i], check, ax_nstd, n_std=n_std,label='generated', edgecolor='firebrick', linestyle='--')
                    _, ellipse2 = confidence_ellipse_2(test_real[:,j], test_real[:,i], check, ax_nstd, n_std=n_std, label='real', edgecolor='blue', linestyle=':')

                else:
                    ellipse1 = confidence_ellipse_2(test_gen[:,j], test_gen[:,i], check, ax_nstd, n_std=n_std,label='generated', edgecolor='firebrick', linestyle='--')
                    ellipse2 = confidence_ellipse_2(test_real[:,j], test_real[:,i], check, ax_nstd, n_std=n_std, label='real', edgecolor='blue', linestyle=':')

                vertices1 = ellipse1.get_verts()
                vertices2 = ellipse2.get_verts()

                ellipse1 = Polygon(vertices1)
                ellipse2 = Polygon(vertices2)

                intersect = ellipse1.intersection(ellipse2)
                union = ellipse1.union(ellipse2)

                area = intersect.area/union.area
                iou_list.append(area)

                if check == True:
                    ax_nstd.set_title("[{}] data: {} #{} iou : {:.4f} emd : {:.4f}".format(model, datatype, factor, area, test_emd_list[cnt]))
                    ax_nstd.set_xlim([min_x_axis-8, max_x_axis+8])
                    ax_nstd.set_ylim([min_y_axis-8, max_y_axis+8])
                    ax_nstd.set_xlabel(label[i])
                    ax_nstd.set_ylabel(label[j])
                    ax_nstd.legend()
                    plt.show

                cnt +=1
    return np.array(iou_list)

def ellipsoid_iou(test_gen, test_real, test_emd_list, n_std, datatype, train_Y_mean, train_Y_std, model, check=False):

    result = []

    test_gen = (test_gen-train_Y_mean)/train_Y_std
    test_real = (test_real-train_Y_mean)/train_Y_std

    num_of_cycle = test_gen.shape[0]
    sample_num = test_gen.shape[1]

    for factor in range(num_of_cycle):
        factor_iou = ellipsoid_2d_2(test_gen[factor], test_real[factor], test_emd_list[factor], 4.605, datatype, factor+1, model, check)
        result.append(factor_iou)

    return np.array(result)

def new_EMD_all_pair_each_X_integral(generated_samples, real_samples, real_bin_num, num_of_cycle, min_list, max_list, train_mean, train_std, minmax, check=False): #여러 X에 대해 각각 쪼개서, 모든 pair(36가지)에 대한 EMD list를 모아서 뱉는 함수.
    dim = real_samples.shape[1]
    
    EMD_score_list = []
    sink_score_list = []
    
    xy_mean = train_mean
    xy_std = train_std
    
    for factor in range(num_of_cycle):
        
        print("Evaluating EMD for factor :", factor, "...")
        
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

def EMD_test_v4(gen_samples, real_samples, index_x, index_y, x_max, y_max, x_min, y_min, real_min, real_max, real_bin_len, check=False, real_bin_num=10):
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

def EMD_test_v5(gen_samples, real_samples, index, sample_max, sample_min, real_min, real_max, real_bin_len, check=False, real_bin_num=10):
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

def factor_wise_EMD_global(generated_samples_cycle, real_samples_cycle, min_list_cycle, max_list_cycle, train_Y_mean, train_Y_std, check=False, real_bin_num=10):
    
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
            #print('Dimentsion (i,j): ', i,j)
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






def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


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


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def load_samples_and_calculate_EMD(filepath, real_bin_num, minmax, real_samples, min_list, max_list, train_mean, train_std):
    with (open(filepath, "rb")) as openfile:
        result = pickle.load(openfile)
    test_gen = result['test sample']
    
    num_of_cycle = test_gen.shape[0]
    num_in_cycle = real_samples.shape[1]
    test_gen_sample_num = test_gen.shape[1]

    ###################### Calculate EMD ######################
    try : 
        aa = result['test EMD']
    except :
        test_EMD_score_list, test_sink_score_list = new_EMD_all_pair_each_X_integral(generated_samples = test_gen, real_samples = real_samples, real_bin_num=real_bin_num, num_of_cycle=num_of_cycle, min_list = min_list, max_list = max_list, train_mean=train_mean, train_std = train_std, minmax=minmax, check=False) 
        
        ###################### Add 'EMD value' to file #####################
        result['test EMD'] = test_EMD_score_list
        with (open(filepath, "wb")) as openfile:
            pickle.dump(result, openfile)
    
    return result


def load_samples_and_calculate_MMD(filepath, real_samples, min_list, max_list, gamma=100.0):
    with (open(filepath, "rb")) as openfile:
        result = pickle.load(openfile)
    test_gen = result['test sample']
    
    num_of_cycle = test_gen.shape[0]
    num_in_cycle = real_samples.shape[1]
    test_gen_sample_num = test_gen.shape[1]
    
    ###################### Calculate MMD ######################
    try :
        aa = result['test MMD']
    except :
        test_MMD_score_list = []
        for i in range(len(real_samples)):
            test_gen_tmp = (test_gen[i] - min_list)/max_list
            real_samples_tmp = (real_samples[i] - min_list)/max_list
            test_MMD_score_list.append(mmd_rbf(test_gen_tmp, real_samples_tmp, gamma=gamma))    

        ###################### Add 'MMD value' to file #####################
        result['test MMD'] = np.array(test_MMD_score_list)
        with (open(filepath, "wb")) as openfile:
            pickle.dump(result, openfile)
    
    return result

# result = ellipsoid_iou(test_gen, test_real, test_emd_list, 4.605, datatype, train_Y_mean, train_Y_std, model)

def load_samples_and_calculate_iou(filepath, real_samples, datatype, train_Y_mean, train_Y_std, model):
    with (open(filepath, "rb")) as openfile:
        result = pickle.load(openfile)
    test_gen = result['test sample']
    test_emd_list = result['test EMD']
    
    num_of_cycle = test_gen.shape[0]
    num_in_cycle = real_samples.shape[1]
    test_gen_sample_num = test_gen.shape[1]
    
    ###################### Calculate MMD ######################
    try :
        aa = result['test IoU']
    except :
        iou_list = ellipsoid_iou(test_gen, real_samples, test_emd_list, 4.605, datatype, train_Y_mean, train_Y_std, model, check=False)

        ###################### Add 'MMD value' to file #####################
        result['test IoU'] = np.array(iou_list)
        with (open(filepath, "wb")) as openfile:
            pickle.dump(result, openfile)
    
    return result
