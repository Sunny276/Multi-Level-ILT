import os
import numpy as np
import torch
import torch.fft as fft

def read_kernel(kernel_file = 'kernel.txt', scale_file = 'scales.txt'):
    dir_path = "./litho"
    with open(os.path.join(dir_path, kernel_file)) as f:
        data = f.readlines()
    data = [list(map(float, line.split())) for line in data]
    data = [[line[i]+line[i+1]*1j for i in range(0,69,2)] for line in data]
    kernels = np.array([np.array(data[35*i:35*(i+1)]).T for i in range(24)])
    
    with open(os.path.join(dir_path, scale_file)) as f:
        scales = f.readlines()[1:]
    scales = np.array(list(map(float, scales)))
    return kernels, scales

class Lithosimer(object):
    def __init__(self, device='cpu'):
        kernels, scales = read_kernel()
        self.kernels = torch.tensor(kernels, requires_grad=False).to(device)
        self.scales = torch.tensor(scales, requires_grad=False).to(device)
        
        ### defocus kernels ans scales ###
        kernels, scales = read_kernel(kernel_file='def_kernel.txt', scale_file='def_scales.txt')
        self.def_kernels = torch.tensor(kernels, requires_grad=False).to(device)
        self.def_scales = torch.tensor(scales, requires_grad=False).to(device)
        
    def litho_show(self, img, dose=1.0, alpha=50, thr=0.225, flag_defocus=False, ilt_option=True):
        img = img * dose
        img_fft = fft.fft2(img)
        img_fft = fft.fftshift(img_fft)
        
        input_size = img.size(1)
        ctr = int(input_size/2)
        d2 = int(self.kernels.size(1) / 2)
        d3 = d2
        if(self.kernels.size(1) % 2 != 0):
            d3 += 1
        
        img_fft = img_fft[ctr-d2:ctr+d3, ctr-d2:ctr+d3]
        
        if flag_defocus:
            res = torch.mul(self.def_kernels, img_fft)
            res = torch.abs(fft.ifft2(res, (input_size, input_size)))  
            res = torch.mul(self.def_scales.unsqueeze(1).unsqueeze(2), torch.square(res))
        else:
            res = torch.mul(self.kernels, img_fft)
            res = torch.abs(fft.ifft2(res, (input_size, input_size)))  
            res = torch.mul(self.scales.unsqueeze(1).unsqueeze(2), torch.square(res))
            
        res = torch.sum(res, 0)
        
        if ilt_option:
            res = torch.sigmoid(alpha*(res-thr))
        else:
            res[res>=thr] = 1
            res[res<thr] = 0 
             
        return res
        
        
          
        
        
        
        