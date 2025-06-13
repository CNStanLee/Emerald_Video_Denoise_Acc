import pytorch_nndct
import torch
from pytorch_nndct.apis import Inspector
import torch.nn as nn




class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        # downrelu = nn.LeakyReLU(0.2, inplace=True)
        downrelu = nn.LeakyReLU(0.1, inplace=True)
        uprelu = nn.ReLU(inplace=True)
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.ReLU()]  # Output activation
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv, nn.Identity(), uprelu, upconv, nn.Identity()]
        else:
            model = [downrelu, downconv, nn.Identity(), submodule, uprelu, upconv, nn.Identity()]
            if use_dropout:
                model += [nn.Identity()]  # Replace dropout with Identity

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs):
        super(UnetGenerator, self).__init__()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock(512, 512, innermost=True)
        for _ in range(num_downs - 5):  # 5 includes innermost + 4 outer layers
            unet_block = UnetSkipConnectionBlock(512, 512, submodule=unet_block, use_dropout=True)
        unet_block = UnetSkipConnectionBlock(256, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(128, 256, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 128, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, 64, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)
    

model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8)

# load the model weights
model_weights = 'training_pix2pix_denoiser_denoiser_f32.pth'
model.load_state_dict(torch.load(model_weights, map_location='cpu'))

# U50 DPUCAHX8H
inspector = Inspector("DPUCADF8H_ISA0") # by target name
input = torch.randn([1, 3, 256, 256])
device='cpu'
inspector.inspect(model, input)