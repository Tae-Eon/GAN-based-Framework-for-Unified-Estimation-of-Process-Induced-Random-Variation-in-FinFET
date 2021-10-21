from arguments import get_args

import torch
import data_handler
import networks
import trainer
import utils
import numpy as np
import sample_data.sample_utils as sample_utils
import pickle

import os, time
import scipy.io as sio
from torch.optim import lr_scheduler


# Arguments
args = get_args()

log_name = 'naive_date_{}_data_{}_model_{}_{}_seed_{}_lr_{}_{}_{}_hidden_dim_{}_{}_batch_size_{}_noise_d_{}_sample_num_{}_tr_num_in_cycle_{}_layer_{}'.format(
    args.date,
    args.dataset,
    args.mean_model_type,
    args.gan_model_type,    
    args.seed,
    args.mean_lr,  
    args.g_lr,      
    args.d_lr,        
    args.mean_hidden_dim,
    args.gan_hidden_dim,
    args.batch_size,  
    args.noise_d,
    args.sample_num, args.tr_num_in_cycle, args.layer
)

utils.set_seed(args.seed)

# # Mean model architecture ( naming for training & sampling )
# mean_model_spec = 'date_{}_data_{}_batch_{}_model_{}_lr_{}_tr_num_in_cycle_{}'.format(args.date, args.dataset, args.batch_size, args.mean_model_type, args.mean_lr, args.tr_num_in_cycle)

# gan model architecture ( naming for training & sampling )
gan_model_spec = 'naive_date_{}_data_{}_batch_{}_model_{}_noise_d_{}_hidden_dim_{}_layer_{}_lr_g_{}_d_{}_tr_num_in_cycle_{}_seed_{}'.format(args.date, args.dataset, args.batch_size, args.gan_model_type, args.noise_d, args.gan_hidden_dim, args.layer, args.g_lr, args.d_lr, args.tr_num_in_cycle, args.seed)

if args.pdrop is not None:
    gan_model_spec += '_pdrop_{}'.format(args.pdrop)
    log_name += '_pdrop_{}'.format(args.pdrop)
    
if args.clipping is not None:
    gan_model_spec += '_clipping_{}'.format(args.clipping)
    log_name += '_clipping_{}'.format(args.clipping)
    

print(log_name)

print("="*100)
print("Arguments =")
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print("="*100)

# Dataset
dataset = data_handler.DatasetFactory.get_dataset(args)

# Test specific dataset
dataset_test = data_handler.DatasetFactory.get_test_dataset(args)
# print(dataset_test)

# loss result
kwargs = {'num_workers': args.workers}

# print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")
torch.set_default_tensor_type('torch.cuda.FloatTensor')



### 상수설정

print(dataset.train_X)
X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(args, dataset.train_X, dataset.train_Y) #

train_Y_min = np.min(dataset.train_Y, axis=0)
train_Y_max = np.max(dataset.train_Y, axis=0)

minmax = 'train_real_global'

print("@@@@@@@debug@@@@@@@@", X_train_mean, X_train_std, Y_train_mean, Y_train_std)

print(" Assign mean, std for Training data ")
print("X train mean, std", X_train_mean, X_train_std) #
print("Y train mean, std", Y_train_mean, Y_train_std) #

print(" Y min, Y max for EMD ")
print("Y min", train_Y_min) #
print("Y max", train_Y_max) #

train_dataset_loader = data_handler.SemiLoader(args, dataset.train_X, 
                                                     dataset.train_Y, 
                                                     X_train_mean, X_train_std, Y_train_mean, Y_train_std) #

val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X_per_cycle, 
                                                    dataset.val_Y_per_cycle, 
                                                    X_train_mean, X_train_std, Y_train_mean, Y_train_std)

test_dataset_loader = data_handler.SemiLoader(args, dataset_test.test_X_per_cycle, 
                                                       dataset_test.test_Y_per_cycle, 
                                                       X_train_mean, X_train_std, Y_train_mean, Y_train_std)
    


# Dataloader

train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs) #

val_iterator = torch.utils.data.DataLoader(val_dataset_loader, batch_size=1, shuffle=True, **kwargs)

test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=1, shuffle=False)

# model

generator, discriminator = networks.ModelFactory.get_gan_model(args)

# weight initiailzation

generator.apply(utils.init_params)
discriminator.apply(utils.init_params)

generator.cuda()
discriminator.cuda()

print(generator, discriminator)

# scheduler

optimizer_g = torch.optim.Adam(generator.parameters(), lr = args.g_lr)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = args.d_lr)

exp_gan_lr_scheduler = lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)

if args.gan_model_type == 'gan1' or 'wgan' or 'gan2' or 'gan3' or 'gan4' or 'wgan_gp':
    testType = 'naive_gan'

print(testType)
t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, args.num_of_output, testType)

# trainer

gan_mytrainer = trainer.TrainerFactory.get_gan_trainer(train_iterator, val_iterator, generator, discriminator, args, optimizer_g, optimizer_d, exp_gan_lr_scheduler) #

# ====== TRAIN ======

if args.mode == 'train' and not os.path.isfile('./models/generator/'+gan_model_spec):
    
    t_start = time.time()
    
    for epoch in range(args.gan_nepochs):

        gan_mytrainer.train()
        p_real, p_fake = gan_mytrainer.evaluate()
        current_d_lr = gan_mytrainer.current_d_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr_d:{:.6f}, || p_real:{:.6f}, p_fake:{:.6f}".format(epoch, current_d_lr, p_real, p_fake))
            
    t_end = time.time()
    # net.state_dict()
    torch.save(generator.state_dict(), './models/generator/'+gan_model_spec)
    torch.save(discriminator.state_dict(), './models/discriminator/'+gan_model_spec)
else:
    print()
    print('Load mean model----------------')
    print()
    gan_mytrainer.G.load_state_dict(torch.load('./models/generator/'+gan_model_spec))


# ====== EVAL ======
result = {}

# Validation set
val_total_result, val_total_num = t_classifier.sample(generator, Y_train_mean, Y_train_std, val_iterator, args.num_of_input+args.one_hot, args.num_of_output, args.noise_d)

# val emd

num_of_cycle = dataset.val_Y_per_cycle.shape[0]
num_in_cycle = int(dataset.val_Y.shape[0]/num_of_cycle)
print(num_of_cycle, num_in_cycle)
val_total_result = val_total_result.reshape(num_of_cycle, args.sample_num, -1)
val_real = dataset.val_Y.reshape(num_of_cycle, num_in_cycle, -1)

val_EMD_score_list, val_sink_score_list = sample_utils.new_EMD_all_pair_each_X_integral(generated_samples = val_total_result, real_samples = val_real, real_bin_num=args.real_bin_num, num_of_cycle=num_of_cycle, min_list = train_Y_min, max_list = train_Y_max, train_mean=Y_train_mean, train_std = Y_train_std, minmax=minmax, check=False) 

# Test set
test_total_result, test_total_num = t_classifier.sample(generator, Y_train_mean, Y_train_std, test_iterator, args.num_of_input+args.one_hot, args.num_of_output, args.noise_d)

# test emd
num_of_cycle = dataset_test.test_Y_per_cycle.shape[0]
num_in_cycle = int(dataset_test.test_Y.shape[0]/num_of_cycle)
print(num_of_cycle, num_in_cycle)

test_total_result = test_total_result.reshape(num_of_cycle, args.sample_num, -1)
test_real = dataset_test.test_Y.reshape(num_of_cycle, num_in_cycle, -1)

test_EMD_score_list, test_sink_score_list = sample_utils.new_EMD_all_pair_each_X_integral(generated_samples = test_total_result, real_samples = test_real, real_bin_num=args.real_bin_num, num_of_cycle=num_of_cycle, min_list = train_Y_min, max_list = train_Y_max, train_mean=Y_train_mean, train_std = Y_train_std, minmax=minmax, check=False) 

result['validation sample'] = val_total_result
result['validation EMD'] = val_EMD_score_list
result['test sample'] = test_total_result
result['test EMD'] = test_EMD_score_list

# # 3: num_of_input
# # 6: num_of_output
    
path = log_name + '.pkl'
with open('sample_data/' + path, "wb") as f:
    pickle.dump(result, f)