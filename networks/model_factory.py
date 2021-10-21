import torch

class ModelFactory():
    def __init__(self):
        pass
    
    @staticmethod
    def get_mean_model(args):

        if 'mlp' in args.mean_model_type:
            print('sk')
            import networks.mean_mlp as mean_mlp
            return mean_mlp.Net(mean_hidden_dim=args.mean_hidden_dim, num_of_input=args.num_of_input, num_of_output=args.num_of_output)
            
    def get_gan_model(args):
        
        num_of_input = args.num_of_input
        one_hot = args.one_hot
        
        if args.gan_model_type == 'gan1':
            
            import networks.gan1 as gan
            return gan.gen1(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.dis1(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'gan2':
            
            import networks.gan2 as gan
            return gan.gen2(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.dis2(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim, args.pdrop)
        
        elif args.gan_model_type == 'gan3':
            
            import networks.gan3 as gan
            return gan.gen3(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output, args.pdrop), gan.dis3(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim, args.pdrop)
        
        elif args.gan_model_type == 'gan_deep':
            
            import networks.gan_deep as gan
            return gan.gen4(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.dis4(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'wgan':
            
            import networks.wgan as gan
            return gan.wgan_gen(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.wgan_dis(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'gan4':
            print("what")
            import networks.gan4 as gan
            return gan.gen4(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.dis4(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim, args.pdrop)
        
        elif args.gan_model_type == 'gan5':
            import networks.gan5 as gan
            return gan.gen5(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.dis5(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim, args.pdrop)
        
        elif args.gan_model_type == 'gan6':
            import networks.gan6 as gan
            return gan.gen6(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.dis6(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'wgan':
            import networks.wgan as gan
            return gan.wgan_gen(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.wgan_dis(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'wgan2':
            import networks.wgan2 as gan
            return gan.wgan_gen2(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.wgan_dis2(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim, args.pdrop)
        
        elif args.gan_model_type == 'wgan3':
            import networks.wgan3 as gan
            return gan.wgan_gen3(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.wgan_dis3(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim)
        
        elif args.gan_model_type == 'wgan4':
            import networks.wgan4 as gan
            return gan.wgan_gen4(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.wgan_dis4(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim, args.pdrop)
        
        elif args.gan_model_type == 'ccgan':
            import networks.ccgan as gan
            return gan.ccgen(args.noise_d+num_of_input+one_hot, args.gan_hidden_dim, args.num_of_output), gan.ccdis(args.num_of_output+num_of_input+one_hot, args.gan_hidden_dim) ## naive
            
    
    def get_vae_model(args):
        
        num_of_input = args.num_of_input
        num_of_output = args.num_of_output
        hidden_dim = args.vae_hidden_dim
        noise_d = args.noise_d
        one_hot = args.one_hot
        
        if args.data_type == 'n' or args.data_type == 'p':
            num_of_input += 1
        elif args.data_type == 'none':
            pass 
        else:
            num_of_input += 2
        
        if args.gan_model_type == 'vae1':
            
            import networks.vae1 as gan
            return gan.VAE(noise_d, hidden_dim, num_of_output, num_of_input+one_hot)

        
    def get_gaussian_model(args):
        
        num_of_input = args.num_of_input
        num_of_output = ((args.num_of_output)**2 - args.num_of_output)//2 + args.num_of_output*2 #mean+cov+diagonal
        one_hot = arge.one_hot
            
        if args.trainer == 'linear_gaussian':
            
            import networks.linear_gaussian as gaussian
            return gaussian.Net(num_of_input=num_of_input+one_hot, num_of_output=num_of_output) # activation 함수가 없음
        
        elif args.trainer == 'mlp_gaussian':
            num_of_hidden = args.mean_hidden_dim
            layer = args.layer
            
            import networks.mlp_gaussian as gaussian
            return gaussian.Net(num_of_hidden, layer, num_of_input=num_of_input+one_hot, num_of_output=num_of_output) # activation 함수가 없음