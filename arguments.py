import argparse

def get_args(): 
    parser = argparse.ArgumentParser(description='SEMI')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--dataset', default='2020_LER_20201008_V008.xlsx', type=str,
                        choices=['2020_LER_20201008_V008.xlsx',
                                 'rdfwfv_train2020_RDFWFV_20201222_V10.xlsx',
                                 'wfv_train2020_RDFWFV_20201222_V10.xlsx',
                                 'rdf_train2020_RDFWFV_20201222_V10.xlsx',
                                 'rdfwfv_wfv_rdf_train2020_RDFWFV_20201222_V10.xlsx',
                                 'train_LERRDFWFV_167set+Testdataset_4set_V002.xlsx',
                                 'updated_train_LERRDFWFV_167set+Testdataset_4set_V002.xlsx'
                                 ],
                        help='(default=%(default)s)')
                                 
    parser.add_argument('--dataset_test', default='2020_LER_20201102_testset_V04.xlsx', type=str,
                        choices=['2020_LER_20201102_testset_V04.xlsx',
                                 'rdfwfv_wfv_rdf_test2020_RDFWFV_20201222_V10.xlsx',
                                 '2021_RDFWFV_20210107.xlsx',
                                 'test_LERRDFWFV_167set+Testdataset_4set_V002.xlsx',
                                 'updated_test_LERRDFWFV_167set+Testdataset_4set_V002.xlsx'
                                ],
                        help='(default=%(default)s)')
    parser.add_argument('--trainer', type=str, required=True, 
                        choices=['gan',
                                 'wgan',
                                 'mlp_gaussian'
                                ])
    parser.add_argument('--model_type', required=False, type=str,
                        choices=['linear_gaussian','mlp_gaussian'], 
                        help='(default=%(default)s)')
    parser.add_argument('--gan_model_type', default=False, type=str, required=False,
                        choices=['gan1', 'ccgan'], 
                        help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate for gaussian (default: 5e-5. Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001. Note that g_lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--d_lr', type=float, default=0.0005,
                        help='learning rate (default: 0.0005. Note that d_lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--noise_d', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--mean_hidden_dim', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--gan_hidden_dim', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--gaussian_nepochs', type=int, default=1000, help='Number of epochs for each gaussian')
    parser.add_argument('--mean_nepochs', type=int, default=1000, help='Number of epochs for each mean increment')
    parser.add_argument('--gan_nepochs', type=int, default=200, help='Number of epochs for each gan increment')    
    parser.add_argument('--workers', type=int, default=0, help='Number of workers in Dataloaders')
    parser.add_argument('--num_of_input', type=int, required=True, help='Number of input for data')
    parser.add_argument('--num_of_output', type=int, default=6, help='Number of output for data')
    parser.add_argument('--sample_num', type=int, default=50, help='sampling number')
    parser.add_argument('--pdrop', type=float, default=None, help='dropout rate')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--tr_num_in_cycle', type=int, default=50, help='number of cycle in tr_X')
    parser.add_argument('--clipping', default=None, type=float, help='weight clipping for wgan')
    parser.add_argument('--real_bin_num', default=10, type=int, help='EMD bin number')
    parser.add_argument('--layer', default=1, type=int, help='number of multi layer')
    parser.add_argument('--one_hot', default=0, type=int, help='one hot input')
    
    ######ccgan######
    parser.add_argument('--threshold_type', type=str, default='hard',
                        choices=['soft', 'hard'])
    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--kappa', type=float, default=-1.0)
    parser.add_argument('--result_path', type=str, default='./sample_data')
    parser.add_argument('--discriminator_path', type=str, default=None)
    parser.add_argument('--generator_path', type=str, default=None)
    parser.add_argument('--fix_discriminator', default=False, action='store_true', help='fix discriminator when action')
    parser.add_argument('--fix_generator', default=False, action='store_true', help='fix discriminator when action')

    

#     parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()

    return args
