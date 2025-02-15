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
result = {}
result['train_prob'] = []
result['test_prob'] = []
result['test_mmd_log'] = []


log_name = 'naive_date_{}_data_{}_model_{}_seed_{}_lr_{}_{}_hidden_dim_{}_batch_size_{}_noise_d_{}_sample_num_{}_tr_num_in_cycle_{}_layer_{}'.format(
    args.date,
    args.dataset,
    args.gan_model_type,    
    args.seed,
    args.g_lr,      
    args.d_lr,        
    args.gan_hidden_dim,
    args.batch_size,  
    args.noise_d,
    args.sample_num, args.tr_num_in_cycle, args.layer
)

utils.set_seed(args.seed)
utils.makedirs('models/discriminator')
utils.makedirs('models/generator')
utils.makedirs(args.result_path)

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
    
if args.gan_model_type=='ccgan':
    gan_model_spec += '_kappa_'+str(args.kappa)+'_kernel_sigma_'+str(args.kernel_sigma)
    log_name += '_kappa_'+str(args.kappa)+'_kernel_sigma_'+str(args.kernel_sigma)
    
if args.fix_discriminator:
    gan_model_spec += str(args.fix_discriminator)
    log_name += str(args.fix_discriminator)
if args.fix_generator:
    gan_model_spec += str(args.fix_generator)
    log_name += str(args.fix_discriminator)
if args.gan_nepochs != 200 :
    gan_model_spec += '_epochs_{}'.format(args.gan_nepochs)
    log_name += '_epochs_{}'.format(args.gan_nepochs)

print('log_name :', log_name)

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

#print(torch.cuda.device_count())
print("GPU availiable: ", torch.cuda.is_available())
print("Let's use", torch.cuda.device_count(), "GPUs!")

    
print("Inits...")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')



### 상수설정

train_Y_min = np.min(dataset.train_Y, axis=0)
train_Y_max = np.max(dataset.train_Y, axis=0)

minmax = 'train_real_global'

print(" Y min, Y max for EMD ")
print("Y min", train_Y_min) #
print("Y max", train_Y_max) #

train_dataset_loader = data_handler.SemiLoader(args, dataset.train_X, 
                                                     dataset.train_Y, 
                                                     dataset.train_X_mean, dataset.train_X_std, dataset.train_Y_mean, dataset.train_Y_std) #

# val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X_per_cycle, 
#                                                     dataset.val_Y_per_cycle, 
#                                                     dataset.train_X_mean, dataset.train_X_std, dataset.train_Y_mean, dataset.train_Y_std)

test_eval_dataset_loader = data_handler.SemiLoader(args, dataset_test.test_X,
                                                       dataset_test.test_Y,
                                                       dataset.train_X_mean, dataset.train_X_std, dataset.train_Y_mean, dataset.train_Y_std)

test_dataset_loader = data_handler.SemiLoader(args, dataset_test.test_X_per_cycle, 
                                                       dataset_test.test_Y_per_cycle, 
                                                       dataset.train_X_mean, dataset.train_X_std, dataset.train_Y_mean, dataset.train_Y_std)
    


# Dataloader

train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs) #

# val_iterator = torch.utils.data.DataLoader(val_dataset_loader, batch_size=1, shuffle=True, **kwargs)
test_eval_iterator = torch.utils.data.DataLoader(test_eval_dataset_loader, batch_size=args.batch_size, shuffle=False)

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

if args.gan_model_type == 'gan1' or 'ccgan':
    testType = 'naive_gan'

print(testType)
t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, args.num_of_output, testType)

# trainer

gan_mytrainer = trainer.TrainerFactory.get_gan_trainer(train_iterator, test_eval_iterator, generator, discriminator, args, optimizer_g, optimizer_d, exp_gan_lr_scheduler) #

# ====== TRAIN ======

sample_seed_num = 10
num_of_cycle = dataset_test.test_Y_per_cycle.shape[0]
num_in_cycle = int(dataset_test.test_Y.shape[0]/num_of_cycle)
print(num_of_cycle, num_in_cycle)
test_real = dataset_test.test_Y.reshape(num_of_cycle, num_in_cycle, -1)

if args.generator_path is not None :
    gan_mytrainer.G.load_state_dict(torch.load(args.generator_path))
if args.discriminator_path is not None :
    gan_mytrainer.D.load_state_dict(torch.load(args.discriminator_path))

if args.mode == 'train' :
    
    #t_start = time.time()
    
    for epoch in range(args.gan_nepochs):

        gan_mytrainer.train()
#         tr_p_real, tr_p_fake = gan_mytrainer.evaluate(mode='train')
#         t_p_real, t_p_fake = gan_mytrainer.evaluate(mode='test')
#         current_d_lr = gan_mytrainer.current_d_lr

#         if((epoch+1)% 1 == 0):
#             print("epoch:{:2d}, lr_d:{:.6f}, || train || p_real:{:.6f}, p_fake:{:.6f}".format(epoch, current_d_lr, tr_p_real, tr_p_fake))
#             print("epoch:{:2d}, lr_d:{:.6f}, || test || p_real:{:.6f}, p_fake:{:.6f}".format(epoch, current_d_lr, t_p_real, t_p_fake))
#         result['train_prob'].append((tr_p_real, tr_p_fake))
#         result['test_prob'].append((t_p_real, t_p_fake))
        if((epoch+1)% 5 == 0):
            if args.record_mmd == True:            
                test_total_result_tmp, _ = t_classifier.sample(generator, dataset.train_Y_mean, dataset.train_Y_std, test_iterator, args.num_of_input+args.one_hot, args.num_of_output, args.noise_d, sample_seed_num)
                test_total_result_tmp = test_total_result_tmp.reshape(sample_seed_num, num_of_cycle, args.sample_num, -1)
                gamma_list = np.ones(num_of_cycle)*args.gamma_mmd
                MMDs = utils.calculate_MMD(test_total_result_tmp, test_real, dataset.train_Y_mean, dataset.train_Y_std, gamma_list)
                result['test_mmd_log'].append(np.mean(MMDs))
    #t_end = time.time()
    # net.state_dict()
    torch.save(generator.state_dict(), './models/generator/'+gan_model_spec)
    torch.save(discriminator.state_dict(), './models/discriminator/'+gan_model_spec)
else:
    print()
    print('Load mean model----------------')
    print()
    gan_mytrainer.G.load_state_dict(torch.load('./models/generator/'+gan_model_spec))
    # gan_mytrainer.G.load_state_dict(torch.load(args.generator_path))
    # gan_mytrainer.D.load_state_dict(torch.load(args.discriminator_path))

# ====== EVAL ======

# Validation set
# val_total_result, val_total_num = t_classifier.sample(generator, dataset.train_Y_mean, dataset.train_Y_std, val_iterator, args.num_of_input+args.one_hot, args.num_of_output, args.noise_d)

# val emd
# num_of_cycle = dataset.val_Y_per_cycle.shape[0]
# num_in_cycle = int(dataset.val_Y.shape[0]/num_of_cycle)
# print(num_of_cycle, num_in_cycle)
# val_total_result = val_total_result.reshape(num_of_cycle, args.sample_num, -1)
# val_real = dataset.val_Y.reshape(num_of_cycle, num_in_cycle, -1)

# val_EMD_score_list, val_sink_score_list = sample_utils.new_EMD_all_pair_each_X_integral(generated_samples = val_total_result, real_samples = val_real, real_bin_num=args.real_bin_num, num_of_cycle=num_of_cycle, min_list = train_Y_min, max_list = train_Y_max, train_mean=dataset.train_Y_mean, train_std = dataset.train_Y_std, minmax=minmax, check=False) 

# Test set



test_total_result, test_total_num = t_classifier.sample(generator, dataset.train_Y_mean, dataset.train_Y_std, test_iterator, args.num_of_input+args.one_hot, args.num_of_output, args.noise_d, sample_seed_num)

test_total_result = test_total_result.reshape(sample_seed_num, num_of_cycle, args.sample_num, -1)
print("sampled result of {} seeds: {}".format(sample_seed_num, test_total_result.shape))


# test_EMD_score_list, test_sink_score_list = sample_utils.new_EMD_all_pair_each_X_integral(generated_samples = test_total_result, real_samples = test_real, real_bin_num=args.real_bin_num, num_of_cycle=num_of_cycle, min_list = train_Y_min, max_list = train_Y_max, train_mean=dataset.train_Y_mean, train_std = dataset.train_Y_std, minmax=minmax, check=False) 

result['X_mean'] = dataset.train_X_mean
result['X_std'] = dataset.train_X_std
result['Y_mean'] = dataset.train_Y_mean
result['Y_std'] = dataset.train_Y_std

# result['validation sample'] = val_total_result
#result['validation EMD'] = val_EMD_score_list
result['test sample'] = test_total_result
#result['test EMD'] = test_EMD_score_list
#result['train time'] = t_end-t_start
# # 3: num_of_input
# # 6: num_of_output
    
file_name = log_name + '.pkl'
with open(args.result_path + '/' + file_name, "wb") as f:
    pickle.dump(result, f)
