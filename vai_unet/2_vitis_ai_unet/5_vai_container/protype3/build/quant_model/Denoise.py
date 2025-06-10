# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Denoise(torch.nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Denoise::input_0
        self.module_1 = py_nndct.nn.Module('const') #Denoise::41
        self.module_2 = py_nndct.nn.Module('const') #Denoise::58
        self.module_3 = py_nndct.nn.Module('const') #Denoise::88
        self.module_4 = py_nndct.nn.Module('const') #Denoise::105
        self.module_5 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[encoder]/Conv2d[0]/input.2
        self.module_6 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[encoder]/ReLU[1]/input.3
        self.module_7 = py_nndct.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[encoder]/Conv2d[2]/input.4
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[encoder]/ReLU[3]/input.5
        self.module_9 = py_nndct.nn.Module('shape') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/34
        self.module_10 = py_nndct.nn.Module('tensor') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/35
        self.module_11 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/40
        self.module_12 = py_nndct.nn.Module('mul') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/42
        self.module_13 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/47
        self.module_14 = py_nndct.nn.Module('floor') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/48
        self.module_15 = py_nndct.nn.Int() #Denoise::Denoise/Sequential[decoder]/Upsample[0]/49
        self.module_16 = py_nndct.nn.Module('shape') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/51
        self.module_17 = py_nndct.nn.Module('tensor') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/52
        self.module_18 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/57
        self.module_19 = py_nndct.nn.Module('mul') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/59
        self.module_20 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/64
        self.module_21 = py_nndct.nn.Module('floor') #Denoise::Denoise/Sequential[decoder]/Upsample[0]/65
        self.module_22 = py_nndct.nn.Int() #Denoise::Denoise/Sequential[decoder]/Upsample[0]/66
        self.module_23 = py_nndct.nn.Interpolate() #Denoise::Denoise/Sequential[decoder]/Upsample[0]/input.6
        self.module_24 = py_nndct.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[decoder]/Conv2d[1]/input.7
        self.module_25 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[decoder]/ReLU[2]/input.8
        self.module_26 = py_nndct.nn.Module('shape') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/81
        self.module_27 = py_nndct.nn.Module('tensor') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/82
        self.module_28 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/87
        self.module_29 = py_nndct.nn.Module('mul') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/89
        self.module_30 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/94
        self.module_31 = py_nndct.nn.Module('floor') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/95
        self.module_32 = py_nndct.nn.Int() #Denoise::Denoise/Sequential[decoder]/Upsample[3]/96
        self.module_33 = py_nndct.nn.Module('shape') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/98
        self.module_34 = py_nndct.nn.Module('tensor') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/99
        self.module_35 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/104
        self.module_36 = py_nndct.nn.Module('mul') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/106
        self.module_37 = py_nndct.nn.Module('cast') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/111
        self.module_38 = py_nndct.nn.Module('floor') #Denoise::Denoise/Sequential[decoder]/Upsample[3]/112
        self.module_39 = py_nndct.nn.Int() #Denoise::Denoise/Sequential[decoder]/Upsample[3]/113
        self.module_40 = py_nndct.nn.Interpolate() #Denoise::Denoise/Sequential[decoder]/Upsample[3]/input.9
        self.module_41 = py_nndct.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[decoder]/Conv2d[4]/input.10
        self.module_42 = py_nndct.nn.ReLU(inplace=False) #Denoise::Denoise/Sequential[decoder]/ReLU[5]/input
        self.module_43 = py_nndct.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Denoise::Denoise/Sequential[decoder]/Conv2d[6]/136
        self.module_44 = py_nndct.nn.Sigmoid() #Denoise::Denoise/Sequential[decoder]/Sigmoid[7]/137

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(device='cpu', data=2.0, dtype=torch.float)
        self.output_module_2 = self.module_2(device='cpu', data=2.0, dtype=torch.float)
        self.output_module_3 = self.module_3(device='cpu', data=2.0, dtype=torch.float)
        self.output_module_4 = self.module_4(device='cpu', data=2.0, dtype=torch.float)
        self.output_module_5 = self.module_5(self.output_module_0)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(input=self.output_module_8, dim=2)
        self.output_module_10 = self.module_10(device='cpu', data=self.output_module_9, dtype=torch.int)
        self.output_module_11 = self.module_11(input=self.output_module_10, dtype=torch.float)
        self.output_module_12 = self.module_12(other=self.output_module_1, input=self.output_module_11)
        self.output_module_13 = self.module_13(input=self.output_module_12, dtype=torch.float)
        self.output_module_14 = self.module_14(input=self.output_module_13)
        self.output_module_15 = self.module_15(input=self.output_module_14)
        self.output_module_16 = self.module_16(input=self.output_module_8, dim=3)
        self.output_module_17 = self.module_17(device='cpu', data=self.output_module_16, dtype=torch.int)
        self.output_module_18 = self.module_18(input=self.output_module_17, dtype=torch.float)
        self.output_module_19 = self.module_19(other=self.output_module_2, input=self.output_module_18)
        self.output_module_20 = self.module_20(input=self.output_module_19, dtype=torch.float)
        self.output_module_21 = self.module_21(input=self.output_module_20)
        self.output_module_22 = self.module_22(input=self.output_module_21)
        self.output_module_23 = self.module_23(input=self.output_module_8, size=[self.output_module_15,self.output_module_22], scale_factor=None, mode='nearest')
        self.output_module_24 = self.module_24(self.output_module_23)
        self.output_module_25 = self.module_25(self.output_module_24)
        self.output_module_26 = self.module_26(input=self.output_module_25, dim=2)
        self.output_module_27 = self.module_27(device='cpu', data=self.output_module_26, dtype=torch.int)
        self.output_module_28 = self.module_28(input=self.output_module_27, dtype=torch.float)
        self.output_module_29 = self.module_29(other=self.output_module_3, input=self.output_module_28)
        self.output_module_30 = self.module_30(input=self.output_module_29, dtype=torch.float)
        self.output_module_31 = self.module_31(input=self.output_module_30)
        self.output_module_32 = self.module_32(input=self.output_module_31)
        self.output_module_33 = self.module_33(input=self.output_module_25, dim=3)
        self.output_module_34 = self.module_34(device='cpu', data=self.output_module_33, dtype=torch.int)
        self.output_module_35 = self.module_35(input=self.output_module_34, dtype=torch.float)
        self.output_module_36 = self.module_36(other=self.output_module_4, input=self.output_module_35)
        self.output_module_37 = self.module_37(input=self.output_module_36, dtype=torch.float)
        self.output_module_38 = self.module_38(input=self.output_module_37)
        self.output_module_39 = self.module_39(input=self.output_module_38)
        self.output_module_40 = self.module_40(input=self.output_module_25, size=[self.output_module_32,self.output_module_39], scale_factor=None, mode='nearest')
        self.output_module_41 = self.module_41(self.output_module_40)
        self.output_module_42 = self.module_42(self.output_module_41)
        self.output_module_43 = self.module_43(self.output_module_42)
        self.output_module_44 = self.module_44(self.output_module_43)
        return self.output_module_44
