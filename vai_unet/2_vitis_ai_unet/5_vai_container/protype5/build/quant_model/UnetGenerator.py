# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UnetGenerator(torch.nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UnetGenerator::input_0
        self.module_1 = py_nndct.nn.quant_input() #UnetGenerator::UnetGenerator/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/Conv2d[0]/input.2
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ReLU[0]/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/Conv2d[1]/input.4
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.6
        self.module_7 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.7
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.9
        self.module_10 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.10
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.12
        self.module_13 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.13
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.15
        self.module_16 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.16
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.18
        self.module_19 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.19
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.21
        self.module_22 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.22
        self.module_23 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[2]/209
        self.module_24 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[3]/input.23
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[2]/226
        self.module_27 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/229
        self.module_28 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.25
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/246
        self.module_31 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/249
        self.module_32 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.27
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/266
        self.module_35 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/269
        self.module_36 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.29
        self.module_38 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/286
        self.module_39 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/289
        self.module_40 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.31
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/306
        self.module_43 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/309
        self.module_44 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.33
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/326
        self.module_47 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/329
        self.module_48 = py_nndct.nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ConvTranspose2d[4]/input.35
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ReLU[6]/346
        self.module_51 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Cat[Cat]/349
        self.module_52 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/ConvTranspose2d[2]/359
        self.module_53 = py_nndct.nn.Tanh() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/Tanh[3]/360
        self.module_54 = py_nndct.nn.dequant_output() #UnetGenerator::UnetGenerator/DeQuantStub[dequant_stub]/361

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(input=self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_6 = self.module_6(self.output_module_4)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_9 = self.module_9(self.output_module_7)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_12 = self.module_12(self.output_module_10)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_15 = self.module_15(self.output_module_13)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_18 = self.module_18(self.output_module_16)
        self.output_module_19 = self.module_19(self.output_module_18)
        self.output_module_21 = self.module_21(self.output_module_19)
        self.output_module_22 = self.module_22(self.output_module_21)
        self.output_module_23 = self.module_23(self.output_module_22)
        self.output_module_24 = self.module_24(self.output_module_23)
        self.output_module_26 = self.module_26(self.output_module_24)
        self.output_module_27 = self.module_27(tensors=[self.output_module_21,self.output_module_26], dim=1)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_30 = self.module_30(self.output_module_28)
        self.output_module_31 = self.module_31(tensors=[self.output_module_18,self.output_module_30], dim=1)
        self.output_module_32 = self.module_32(self.output_module_31)
        self.output_module_34 = self.module_34(self.output_module_32)
        self.output_module_35 = self.module_35(tensors=[self.output_module_15,self.output_module_34], dim=1)
        self.output_module_36 = self.module_36(self.output_module_35)
        self.output_module_38 = self.module_38(self.output_module_36)
        self.output_module_39 = self.module_39(tensors=[self.output_module_12,self.output_module_38], dim=1)
        self.output_module_40 = self.module_40(self.output_module_39)
        self.output_module_42 = self.module_42(self.output_module_40)
        self.output_module_43 = self.module_43(tensors=[self.output_module_9,self.output_module_42], dim=1)
        self.output_module_44 = self.module_44(self.output_module_43)
        self.output_module_46 = self.module_46(self.output_module_44)
        self.output_module_47 = self.module_47(tensors=[self.output_module_6,self.output_module_46], dim=1)
        self.output_module_48 = self.module_48(self.output_module_47)
        self.output_module_50 = self.module_50(self.output_module_48)
        self.output_module_51 = self.module_51(tensors=[self.output_module_3,self.output_module_50], dim=1)
        self.output_module_52 = self.module_52(self.output_module_51)
        self.output_module_53 = self.module_53(self.output_module_52)
        self.output_module_54 = self.module_54(input=self.output_module_53)
        return self.output_module_54
