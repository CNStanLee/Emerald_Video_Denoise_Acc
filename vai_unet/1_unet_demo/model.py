import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.orthogonal_(self.conv.weight)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        return x

class TransposeConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(TransposeConv2DLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        nn.init.orthogonal_(self.deconv.weight)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.conv1 = Conv2DLayer(3, 64, 3, 1, 1)
        self.conv2 = Conv2DLayer(64, 64, 3, 2, 1)
        self.conv3 = Conv2DLayer(64, 128, 5, 2, 2)
        self.conv4 = Conv2DLayer(128, 128, 3, 1, 1)
        self.conv5 = Conv2DLayer(128, 256, 5, 2, 2)
        self.conv6 = Conv2DLayer(256, 512, 3, 2, 1)
        
        # Decoder
        self.deconv1 = TransposeConv2DLayer(512, 512, 3, 2, 1) # Adjust padding if needed
        self.conv7 = Conv2DLayer(512 + 256, 256, 3, 1, 1)
        
        self.deconv2 = TransposeConv2DLayer(256, 128, 3, 2, 1) # Adjust padding if needed
        self.conv8 = Conv2DLayer(128 + 128, 128, 5, 1, 2)
        
        self.deconv3 = TransposeConv2DLayer(128, 64, 3, 2, 1)
        self.conv9 = Conv2DLayer(64 + 64, 64, 5, 1, 2)
        
        self.deconv4 = TransposeConv2DLayer(64, 64, 3, 2, 1)
        
        # Final conv layer
        self.final_conv = nn.Conv2d(64 + 64, 3, 3, 1, 1)
        nn.init.orthogonal_(self.final_conv.weight)
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        
        # Decoder with skip connections
        deconv1 = self.deconv1(conv6)
        
        if deconv1.size() != conv5.size():
            deconv1 = F.interpolate(deconv1, size=conv5.size()[2:], mode='bilinear', align_corners=False)
        
        skip1 = torch.cat([deconv1, conv5], dim=1)
        conv7 = self.conv7(skip1)
        
        deconv2 = self.deconv2(conv7)
        
        if deconv2.size() != conv3.size():
            deconv2 = F.interpolate(deconv2, size=conv3.size()[2:], mode='bilinear', align_corners=False)
        
        skip2 = torch.cat([deconv2, conv3], dim=1)
        conv8 = self.conv8(skip2)
        
        deconv3 = self.deconv3(conv8)
        
        if deconv3.size() != conv2.size():
            deconv3 = F.interpolate(deconv3, size=conv2.size()[2:], mode='bilinear', align_corners=False)
        
        skip3 = torch.cat([deconv3, conv2], dim=1)
        conv9 = self.conv9(skip3)
        
        deconv4 = self.deconv4(conv9)
        
        if deconv4.size() != conv1.size():
            deconv4 = F.interpolate(deconv4, size=conv1.size()[2:], mode='bilinear', align_corners=False)
        
        skip4 = torch.cat([deconv4, conv1], dim=1)
        output = torch.sigmoid(self.final_conv(skip4))
        
        return output

# Define the model
autoencoder = AutoEncoder()

summary(autoencoder, (3, 256, 256))
