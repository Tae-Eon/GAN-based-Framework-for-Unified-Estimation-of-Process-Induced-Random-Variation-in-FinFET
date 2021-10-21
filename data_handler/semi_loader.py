import torch
import torch.utils.data as td
import numpy as np


class SemiLoader(td.Dataset):
    def __init__(self, args, data_x, data_y, x_mean, x_std, y_mean, y_std):
        
        # data_x : 7-dim
        # data_y : 6-dim
        # x_mean, x_std : 4-dim
        # y_mean, y_std : 6-dim
        
        self.data_x = data_x
#         print(self.data_x)
        self.data_y = data_y
        
        # normalization
#         print(data_x.shape)
#         print(data_x[:,args.num_of_input-3:])
#         print(data_x[:,args.num_of_input-3:].shape)
        if args.one_hot > 0:
            print('semi_loader/data_type', data_x[:,args.num_of_input:].shape)
            data_type = data_x[:,args.num_of_input:]
            print('semi_loader/data_type', data_type.shape)
            temp_x = (data_x[:,:args.num_of_input] - x_mean) / x_std
            temp_y = (data_y - y_mean) / y_std

            temp_x = np.hstack((temp_x, data_type))
            self.data_x = temp_x
            self.data_y = temp_y
        else:
            self.data_x = (data_x - x_mean) / x_std
            self.data_y = (data_y - y_mean) / y_std

#         print(self.data_x)
        
    def __len__(self):
        return len(self.data_x)
           
    def __getitem__(self, index):
        
        x = self.data_x[index]
        y = self.data_y[index]
            
        x = torch.from_numpy(x).float().cuda()
        y = torch.from_numpy(y).float().cuda()
        
        # print(y)
                
        # x[3] = 1.0
        # x[4] = 0.0
        
        return x, y
        