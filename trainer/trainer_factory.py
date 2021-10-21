import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class TrainerFactory():
    def __init__(self):
        pass
    
    # GAN trainer    
    @staticmethod  
    def get_gan_trainer(noise_trainer_iterator, noise_val_iterator, generator, discriminator, args, optimizer_g, optimizer_d, exp_gan_lr_scheduler):
        if args.gan_model_type == 'gan1' or args.gan_model_type == 'gan2' or args.gan_model_type == 'gan3' or args.gan_model_type == 'gan4' or args.gan_model_type == 'gan_deep' or args.gan_model_type == 'gan5' or args.gan_model_type =='gan6':
            import trainer.gan1 as trainer
            
            return trainer.GanTrainer(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, args.noise_d)
        
        elif 'wgan' in args.gan_model_type: 
            import trainer.wgan as trainer
            
            return trainer.GanTrainer(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, args.noise_d, args.clipping)
        
        elif args.gan_model_type == 'wgan_gp':
            import trainer.wgan_gp as trainer
            
            return trainer.GanTrainer(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, args.noise_d, args.gp_strength)
        
        elif args.gan_model_type == 'ccgan': 
            import trainer.ccgan as trainer
            
            return trainer.GanTrainer(noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, args.noise_d, args.clipping, args.kernel_sigma, args.kappa, args.threshold_type)
        
    def get_vae_trainer(noise_trainer_iterator, noise_val_iterator, model, args, optimizer, exp_gan_lr_scheduler):
        if args.gan_model_type == 'vae1':
            import trainer.vae as trainer
            
            return trainer.VAETrainer(noise_trainer_iterator, noise_val_iterator, model, optimizer, exp_gan_lr_scheduler, args.noise_d)
        
        
    # Gaussian trainer
    def get_trainer(train_iterator, val_iterator, model, args, optimizer, exp_lr_scheduler):
        if args.trainer == 'linear_gaussian'or'mlp_gaussian':
            import trainer.mean as trainer
            
            return trainer.MeanTrainer(train_iterator, val_iterator, model, optimizer, exp_lr_scheduler)
        
    

class mean_GenericTrainer:
    """
    Base class for mean trainer
    """
    def __init__(self, train_iterator, val_iterator, mean_model, optimizer, exp_lr_scheduler):
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.model = mean_model
                
        self.optimizer = optimizer
        self.current_lr = None
        
        self.exp_lr_scheduler = exp_lr_scheduler
        
        self.loss = {'train_loss':[], 'val_loss':[]}
        

class gan_GenericTrainer:
    """
    Base class for gan trainer
    """
    def __init__(self, noise_trainer_iterator, noise_val_iterator, generator, discriminator, optimizer_g, optimizer_d, exp_gan_lr_scheduler, noise_d, clipping=None, kernel_sigma=None, kappa=None, threshold_type=None):
        self.train_iterator = noise_trainer_iterator
        self.val_iterator = noise_val_iterator
        
        self.G = generator
        self.D = discriminator
        
        self.optimizer_G = optimizer_g
        self.optimizer_D = optimizer_d
        
        self.exp_gan_lr_scheduler = exp_gan_lr_scheduler
        self.current_d_lr = None
        
        self.noise_d = noise_d
        self.clipping = clipping
        
        self.prob = {'p_real_train':[], 'p_fake_train':[], 'p_real_val':[], 'p_fake_val':[]}
        
        self.kernel_sigma = kernel_sigma
        self.kappa = kappa
        self.threshold_type=threshold_type
        
class vae_GenericTrainer:
    """
    Base class for vae trainer
    """
    def __init__(self, noise_trainer_iterator, noise_val_iterator, model, optimizer, exp_vae_lr_scheduler, noise_d):
        self.train_iterator = noise_trainer_iterator
        self.val_iterator = noise_val_iterator
        
        self.model = model
        
        self.optimizer = optimizer
        
        self.exp_lr_scheduler = exp_vae_lr_scheduler
        self.current_d_lr = None
        
        self.noise_d = noise_d
        
        self.prob = {'loss_train':[], 'loss_val':[]}