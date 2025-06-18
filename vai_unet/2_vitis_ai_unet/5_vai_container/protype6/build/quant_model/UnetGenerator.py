# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UnetGenerator(torch.nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UnetGenerator::input_0
        self.module_1 = py_nndct.nn.quant_input() #UnetGenerator::UnetGenerator/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/Conv2d[0]/input.2
        self.module_3 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/LeakyReLU[0]/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/Conv2d[1]/input.4
        self.module_5 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/LeakyReLU[0]/input.5
        self.module_6 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.6
        self.module_7 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/LeakyReLU[0]/input.7
        self.module_8 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.8
        self.module_9 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/LeakyReLU[0]/input.9
        self.module_10 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.10
        self.module_11 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/LeakyReLU[0]/input.11
        self.module_12 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.12
        self.module_13 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/LeakyReLU[0]/input.13
        self.module_14 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.14
        self.module_15 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/LeakyReLU[0]/input.15
        self.module_16 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/120
        self.module_17 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[3]/input.16
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[4]/132
        self.module_19 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/135
        self.module_20 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.17
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[5]/147
        self.module_22 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/150
        self.module_23 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.18
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[5]/162
        self.module_25 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/165
        self.module_26 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.19
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[5]/177
        self.module_28 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/180
        self.module_29 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.20
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[5]/192
        self.module_31 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/195
        self.module_32 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.21
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[5]/207
        self.module_34 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/210
        self.module_35 = py_nndct.nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ConvTranspose2d[4]/input.22
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ReLU[5]/222
        self.module_37 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Cat[Cat]/225
        self.module_38 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/ConvTranspose2d[2]/input
        self.module_39 = py_nndct.nn.ReLU(inplace=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/ReLU[3]/236
        self.module_40 = py_nndct.nn.dequant_output() #UnetGenerator::UnetGenerator/DeQuantStub[dequant_stub]/237

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(input=self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_14 = self.module_14(self.output_module_13)
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_17 = self.module_17(self.output_module_16)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_19 = self.module_19(dim=1, tensors=[self.output_module_15,self.output_module_18])
        self.output_module_20 = self.module_20(self.output_module_19)
        self.output_module_21 = self.module_21(self.output_module_20)
        self.output_module_22 = self.module_22(dim=1, tensors=[self.output_module_13,self.output_module_21])
        self.output_module_23 = self.module_23(self.output_module_22)
        self.output_module_24 = self.module_24(self.output_module_23)
        self.output_module_25 = self.module_25(dim=1, tensors=[self.output_module_11,self.output_module_24])
        self.output_module_26 = self.module_26(self.output_module_25)
        self.output_module_27 = self.module_27(self.output_module_26)
        self.output_module_28 = self.module_28(dim=1, tensors=[self.output_module_9,self.output_module_27])
        self.output_module_29 = self.module_29(self.output_module_28)
        self.output_module_30 = self.module_30(self.output_module_29)
        self.output_module_31 = self.module_31(dim=1, tensors=[self.output_module_7,self.output_module_30])
        self.output_module_32 = self.module_32(self.output_module_31)
        self.output_module_33 = self.module_33(self.output_module_32)
        self.output_module_34 = self.module_34(dim=1, tensors=[self.output_module_5,self.output_module_33])
        self.output_module_35 = self.module_35(self.output_module_34)
        self.output_module_36 = self.module_36(self.output_module_35)
        self.output_module_37 = self.module_37(dim=1, tensors=[self.output_module_3,self.output_module_36])
        self.output_module_38 = self.module_38(self.output_module_37)
        self.output_module_39 = self.module_39(self.output_module_38)
        self.output_module_40 = self.module_40(input=self.output_module_39)
        return self.output_module_40
