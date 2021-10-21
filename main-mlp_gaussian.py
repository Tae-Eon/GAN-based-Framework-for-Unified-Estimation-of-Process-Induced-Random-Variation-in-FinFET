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

log_name = 'mlp_gaussian_date_{}_data_{}_seed_{}_lr_{}_hidden_{}_{}_batch_size_{}_sample_num_{}_tr_num_in_cycle_{}_epoch_{}'.format(
    args.date,
    args.dataset,
    args.seed,
    args.lr,
    args.layer,
    args.mean_hidden_dim,
    args.batch_size,
    args.sample_num, args.tr_num_in_cycle, args.gaussian_nepochs
)

utils.set_seed(args.seed)

model_spec = 'mlp_gaussian_date_{}_data_{}_batch_{}_lr_{}_hidden_{}_{}_tr_num_in_cycle_{}_epoch_{}'.format(args.date, args.dataset, args.batch_size, args.lr, args.layer, args.mean_hidden_dim, args.tr_num_in_cycle, args.gaussian_nepochs)

# if args.pdrop is not None:
#     model_spec += '_pdrop_{}'.format(args.pdrop)
#     log_name += '_pdrop_{}'.format(args.pdrop)

print(log_name)

print("="*100)
print("Arguments =")
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print("="*100)

# Dataset
dataset = data_handler.DatasetFactory.get_dataset(args) #
 
# Test specific dataset
dataset_test = data_handler.DatasetFactory.get_test_dataset(args) #

# loss result
kwargs = {'num_workers': args.workers}

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
print("Inits...")
torch.set_default_tensor_type('torch.cuda.FloatTensor')



### 상수설정
X_train_mean, X_train_std, Y_train_mean, Y_train_std = utils.train_mean_std(args, dataset.train_X, dataset.train_Y) #

X_train_meancov_mean, X_train_meancov_std, Y_train_meancov_mean, Y_train_meancov_std = utils.train_mean_std(args, dataset.train_X_per_cycle, dataset.train_Y_mean_cov) # 6(mean)+21(cov) ####dataset 만



train_Y_min = np.min(dataset.train_Y, axis=0)
train_Y_max = np.max(dataset.train_Y, axis=0)

minmax = 'train_real_global'


print(" Assign mean, std for Training data ")
print("X train mean, std", X_train_meancov_mean, X_train_meancov_std) #
print("Y train mean, std", Y_train_meancov_mean, Y_train_meancov_std) #

print(" Y min, Y max for EMD ")
print("Y min", train_Y_min) #
print("Y max", train_Y_max) #

train_dataset_loader = data_handler.SemiLoader(args, dataset.train_X_per_cycle, 
                                                     dataset.train_Y_mean_cov, 
                                                     X_train_meancov_mean, X_train_meancov_std, Y_train_meancov_mean, Y_train_meancov_std) #

val_dataset_loader = data_handler.SemiLoader(args, dataset.val_X_per_cycle, 
                                                   dataset.val_Y_mean_cov, 
                                                   X_train_meancov_mean, X_train_meancov_std, Y_train_meancov_mean, Y_train_meancov_std) #

test_X_dataset_loader = data_handler.SemiLoader(args, dataset_test.test_X_per_cycle, 
                                                       dataset_test.test_Y_per_cycle, 
                                                       X_train_meancov_mean, X_train_meancov_std, 0, 1)
    
    
    
# Dataloader
train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, **kwargs) #

val_iterator = torch.utils.data.DataLoader(val_dataset_loader, batch_size=1, shuffle=False, **kwargs) #

test_X_iterator = torch.utils.data.DataLoader(test_X_dataset_loader, batch_size=1, shuffle=False)

# model
model = networks.ModelFactory.get_gaussian_model(args) # mlp gaussian model 만들기


# weight initiailzation
model.apply(utils.init_params)


model.cuda()
print(model)


# scheduler
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=(args.gaussian_nepochs)/5, gamma=0.5)

if args.model_type == 'mlp_gaussian':
    testType = 'gaussian'

t_classifier = trainer.EvaluatorFactory.get_evaluator(args.sample_num, args.num_of_output, testType) #


# trainer
mlp_gaussian_trainer = trainer.TrainerFactory.get_trainer(train_iterator, val_iterator, model, args, optimizer, exp_lr_scheduler)


# ====== TRAIN ======

if args.mode == 'train' and not os.path.isfile('./models/mlp_gaussian/'+model_spec):
    for epoch in range(args.gaussian_nepochs):

        mlp_gaussian_trainer.train()
        val_loss, _ = mlp_gaussian_trainer.evaluate()
        current_lr = mlp_gaussian_trainer.current_lr

        if((epoch+1)% 10 == 0):
            print("epoch:{:2d}, lr_d:{:.6f}, || val_loss:{:.6f}".format(epoch, current_lr, val_loss))
            
    # net.state_dict()
    torch.save(model.state_dict(), './models/mlp_gaussian/'+model_spec)
    
else:
    print()
    print('Load mean model----------------')
    print()
    mlp_gaussian_trainer.model.load_state_dict(torch.load('./models/mlp_gaussian/'+model_spec))


# ====== EVAL ======

result = {}

# Validation set
mean_cov_result, total = t_classifier.mean_cov_sample(model, val_iterator)
    
val_total_result = t_classifier.sample(mean_cov_result, Y_train_meancov_mean, Y_train_meancov_std, total)

# val emd

num_of_cycle = dataset.val_Y_per_cycle.shape[0]
num_in_cycle = int(dataset.val_Y.shape[0]/num_of_cycle)
print(num_of_cycle, num_in_cycle)
val_total_result = val_total_result.reshape(num_of_cycle, args.sample_num, -1)
val_real = dataset.val_Y.reshape(num_of_cycle, num_in_cycle, -1)



val_EMD_score_list, val_sink_score_list = sample_utils.new_EMD_all_pair_each_X_integral(generated_samples = val_total_result, real_samples = val_real, real_bin_num=args.real_bin_num, num_of_cycle=num_of_cycle, min_list = train_Y_min, max_list = train_Y_max, train_mean=Y_train_mean, train_std = Y_train_std, minmax=minmax, check=False) 

# Test set
mean_cov_result, total = t_classifier.mean_cov_sample(model, test_X_iterator)
    
test_total_result = t_classifier.sample(mean_cov_result, Y_train_meancov_mean, Y_train_meancov_std, total)
# test emd
num_of_cycle = dataset_test.test_Y_per_cycle.shape[0]

test_total_result = test_total_result.reshape(num_of_cycle, args.sample_num, -1)

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

    
path = log_name + '.pkl'
with open('sample_data/' + path, "wb") as f:
    pickle.dump(result, f)