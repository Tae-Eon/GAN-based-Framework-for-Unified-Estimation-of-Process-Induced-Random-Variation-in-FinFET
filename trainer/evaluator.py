import torch
import numpy as np
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.utils import _standard_normal


class EvaluatorFactory():
    def __init__(self):
        pass
    
    @staticmethod
    def get_evaluator(sample_num, num_output, testType='gan'):
        if testType == 'gan':
            return gan_evaluator(sample_num)
        elif testType == 'vae':
            return vae_evaluator(sample_num)
        elif testType == 'gaussian':
            return gaussian_evaluator(sample_num, num_output)
        elif testType == 'naive_gan': #
            return naive_gan_evaluator(sample_num) #    

class naive_gan_evaluator():
    
    def __init__(self, sample_num):
        
        self.sample_num = sample_num
        self.prob = {}
        
    def train_eval(self, generator, discriminator, val_loader, noise_d):
        
        p_real, p_fake = 0., 0.
        batch_num = 0
        
        generator.eval()
        discriminator.eval()
        
        for i, data in enumerate(val_loader):
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            z = utils.sample_z(mini_batch_size, noise_d)
            # utils.sample_z
            
            with torch.autograd.no_grad():
                p_real += torch.sum(discriminator(data_y, data_x)/mini_batch_size)
                
                gen_y = generator(z, data_x)
                
                p_fake += torch.sum(discriminator(gen_y, data_x)/mini_batch_size)
                
            batch_num += 1
            
        p_real /= batch_num
        p_fake /= batch_num
        
        self.prob['p_real_val'].append(p_real)
        self.prob['p_fake_val'].append(p_fake)
        
        return p_real, p_fake
        
              
    def sample(self, generator, train_mean, train_std, test_loader, num_of_input, num_of_output, noise_d, sample_seed_num):
        
        result = []
        
        for seed in range(sample_seed_num):
            
            print("Sample generated in seed {}".format(seed))
        
            result_per_seed = []
            total = 0

            generator.eval()

            for i, data in enumerate(test_loader):

                data_x, data_y = data
                print(data_x)
                data_x, data_y = data_x.cuda(), data_y.cuda()

                mini_batch_size = len(data_x)

                with torch.autograd.no_grad():

                    data_x_sample = data_x.repeat(1, self.sample_num).view(-1, num_of_input)

                    # print("batch * sample_num: (expected)", mini_batch_size*self.sample_num, "(result)", data_x_sample.shape)

                    z = utils.sample_z(mini_batch_size*self.sample_num, noise_d)
                    y_hat = generator(z, data_x_sample)

                    y_hat = y_hat.data.cpu().numpy()

                    y_hat = y_hat*train_std + train_mean

                    result_per_seed.append(y_hat)

                    total += mini_batch_size


            result_per_seed = np.array(result_per_seed)
            #noise_result = noise_result.reshape(-1, num_of_output)
            result_per_seed = np.vstack(result_per_seed)
            result.append(result_per_seed)

        result = np.array(result)    
        
        return result, total

    
    
class gaussian_evaluator():
    
    def __init__(self, sample_num, num_output):

        self.sample_num = sample_num
        self.prob = {}
        self.num_output = num_output
        
    def mean_cov_sample(self, best_model, test_iterator):
        
        mean_cov_result = np.array([]).reshape((0,(self.num_output**2-self.num_output)//2 + self.num_output*2))
        total = 0

        best_model.eval()
        
        for i, data in enumerate(test_iterator):
            data_x, data_y = data
            data_x, data_y = data_x.cuda(), data_y.cuda()
            
            mini_batch_size = len(data_x)
            
            with torch.autograd.no_grad():
                y_mean_cov = best_model(data_x)
                
                y_mean_cov = y_mean_cov.data.cpu().numpy()
                
                
                mean_cov_result = np.vstack([mean_cov_result, y_mean_cov])
                
            
                total += mini_batch_size
        
        mean_cov_result = np.array(mean_cov_result)
        #print('final_mean_cov_result',mean_cov_result.shape)   
        

        return mean_cov_result, total
    
    def sample(self, mean_cov_result, Y_train_mean, Y_train_std, total):
        
        mean_cov_result = mean_cov_result*Y_train_std + Y_train_mean
        
        result = np.array([]).reshape((0,self.num_output))
        #mean_cov_result = mean_cov_result*train_std + train_mean
        
        for i in range(total):
            
            temp_result = []
            
            # mean
            mean = mean_cov_result[i,:self.num_output]
            
            # covariance
            cov = np.zeros((self.num_output, self.num_output))
            
            cnt1 = self.num_output*2
            for k in range(1, self.num_output):
                cov[k,:k], cov[:k, k] = mean_cov_result[i, cnt1:cnt1+k], mean_cov_result[i, cnt1:cnt1+k]
                cnt1 += k
                       
            for l in range(self.num_output):
                cov[l,l] = mean_cov_result[i, self.num_output+l]        
            
            
            print('mean', mean)
            print('cov', cov)


            temp_result = np.random.multivariate_normal(mean=mean, cov=cov, size=self.sample_num)
            
            
            #print('temp_result.shpae', temp_result.shape)
            result = np.vstack([result,temp_result])
                
        result = np.array(result)
        
        print("result size: ", result.shape)
        
        return result