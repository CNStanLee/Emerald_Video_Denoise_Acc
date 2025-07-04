import pytorch_nndct.nn as nndct_nn
import pytorch_nndct.nn.modules.functional as functional
import torch
import torch.nn as nn

# import pytorch_nndct.nn as nndct_nn to tell compiler to quantize the relu function


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.Cat = functional.Cat()

        
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.1015625, inplace=True) # Changhong This value is fixed because hardware will compile leakyrelu with a fixed value for efficiency
        uprelu = nn.ReLU(inplace=True)
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            #up = [uprelu, upconv, nn.ReLU()]  # Output activation Changhong, make uprelu followed with Conv
            up = [upconv, nn.ReLU()]  # Changhong, delete uprelu
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            #model = [downrelu, downconv, nn.Identity(), uprelu, upconv, nn.Identity()] # Changhong, make uprelu followed with Conv
            model = [downrelu, downconv, nn.Identity(), upconv , uprelu, nn.Identity()]
        else:
            #model = [downrelu, downconv, nn.Identity(), submodule, uprelu, upconv, nn.Identity()]
            model = [downrelu, downconv, nn.Identity(), submodule, upconv , uprelu, nn.Identity()]# Changhong, make uprelu followed with Conv
            if use_dropout:
                model += [nn.Identity()]  # Replace dropout with Identity

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            #return torch.cat([x, self.model(x)], 1)
            return self.Cat([x, self.model(x)], 1) # Changhong, use pytorch_nndct.nn.modules.functional.Cat instead of torch.cat for quantization compatibility


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs):
        super(UnetGenerator, self).__init__()

        # Changhong: Use nndct_nn.QuantStub and nndct_nn.DeQuantStub for quantization compatibility
        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock(512, 512, innermost=True)
        for _ in range(num_downs - 5):  # 5 includes innermost + 4 outer layers
            unet_block = UnetSkipConnectionBlock(512, 512, submodule=unet_block, use_dropout=True)
        unet_block = UnetSkipConnectionBlock(256, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(128, 256, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 128, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, 64, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        # Changhong: Use nndct_nn.QuantStub and nndct_nn.DeQuantStub for quantization compatibility
        #return self.model(x)
        x = self.quant_stub(x)
        x = self.model(x)
        return self.dequant_stub(x)  # Use dequant_stub to ensure output is compatible


if __name__ == '__main__':
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8)
    print(model)