# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UnetGenerator(torch.nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UnetGenerator::input_0
        self.module_1 = py_nndct.nn.quant_input() #UnetGenerator::UnetGenerator/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/Conv2d[0]/input.3
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ReLU[0]/input.5
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/Conv2d[1]/input.7
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.11
        self.module_6 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.13
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.17
        self.module_8 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.19
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.23
        self.module_10 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.25
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.29
        self.module_12 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.31
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.35
        self.module_14 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.37
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[0]/input.41
        self.module_16 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/Conv2d[1]/input.43
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[2]/5450
        self.module_18 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[3]/input.45
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[2]/5476
        self.module_20 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/5482
        self.module_21 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.49
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/5508
        self.module_23 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/5514
        self.module_24 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.53
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/5540
        self.module_26 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/5546
        self.module_27 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.57
        self.module_28 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/5572
        self.module_29 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/5578
        self.module_30 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.61
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/5604
        self.module_32 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/5610
        self.module_33 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ConvTranspose2d[4]/input.65
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Sequential[model]/ReLU[6]/5636
        self.module_35 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/UnetSkipConnectionBlock[3]/Cat[Cat]/5642
        self.module_36 = py_nndct.nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ConvTranspose2d[4]/input.69
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Sequential[model]/ReLU[6]/5668
        self.module_38 = py_nndct.nn.Cat() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/UnetSkipConnectionBlock[1]/Cat[Cat]/5674
        self.module_39 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/ConvTranspose2d[2]/5693
        self.module_40 = py_nndct.nn.Tanh() #UnetGenerator::UnetGenerator/UnetSkipConnectionBlock[model]/Sequential[model]/Tanh[3]/inputs
        self.module_41 = py_nndct.nn.dequant_output() #UnetGenerator::UnetGenerator/DeQuantStub[dequant_stub]/5695

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_6 = self.module_6(output_module_4)
        output_module_6 = self.module_7(output_module_6)
        output_module_8 = self.module_8(output_module_6)
        output_module_8 = self.module_9(output_module_8)
        output_module_10 = self.module_10(output_module_8)
        output_module_10 = self.module_11(output_module_10)
        output_module_12 = self.module_12(output_module_10)
        output_module_12 = self.module_13(output_module_12)
        output_module_14 = self.module_14(output_module_12)
        output_module_14 = self.module_15(output_module_14)
        output_module_16 = self.module_16(output_module_14)
        output_module_16 = self.module_17(output_module_16)
        output_module_16 = self.module_18(output_module_16)
        output_module_16 = self.module_19(output_module_16)
        output_module_20 = self.module_20(dim=1, tensors=[output_module_14,output_module_16])
        output_module_20 = self.module_21(output_module_20)
        output_module_20 = self.module_22(output_module_20)
        output_module_23 = self.module_23(dim=1, tensors=[output_module_12,output_module_20])
        output_module_23 = self.module_24(output_module_23)
        output_module_23 = self.module_25(output_module_23)
        output_module_26 = self.module_26(dim=1, tensors=[output_module_10,output_module_23])
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_29 = self.module_29(dim=1, tensors=[output_module_8,output_module_26])
        output_module_29 = self.module_30(output_module_29)
        output_module_29 = self.module_31(output_module_29)
        output_module_32 = self.module_32(dim=1, tensors=[output_module_6,output_module_29])
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_35 = self.module_35(dim=1, tensors=[output_module_4,output_module_32])
        output_module_35 = self.module_36(output_module_35)
        output_module_35 = self.module_37(output_module_35)
        output_module_38 = self.module_38(dim=1, tensors=[output_module_0,output_module_35])
        output_module_38 = self.module_39(output_module_38)
        output_module_38 = self.module_40(output_module_38)
        output_module_38 = self.module_41(input=output_module_38)
        return output_module_38
