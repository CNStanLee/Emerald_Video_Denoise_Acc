# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Denoise(torch.nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Denoise::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[encoder]/Conv2d[0]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[encoder]/ReLU[1]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[encoder]/Conv2d[2]/input.4
        self.module_4 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[encoder]/ReLU[3]/32
        self.module_5 = py_nndct.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[1, 1], groups=1, bias=True, dilation=[1, 1]) #Denoise::Denoise/Sequential[decoder]/ConvTranspose2d[0]/input.5
        self.module_6 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[decoder]/ReLU[1]/43
        self.module_7 = py_nndct.nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], output_padding=[1, 1], groups=1, bias=True, dilation=[1, 1]) #Denoise::Denoise/Sequential[decoder]/ConvTranspose2d[2]/input.6
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[decoder]/ReLU[3]/input
        self.module_9 = py_nndct.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[decoder]/Conv2d[4]/64
        self.module_10 = py_nndct.nn.Sigmoid() #Denoise::Denoise/Sequential[decoder]/Sigmoid[5]/65

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        return self.output_module_10
