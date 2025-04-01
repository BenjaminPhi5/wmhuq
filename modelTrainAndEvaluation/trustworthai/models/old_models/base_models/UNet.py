# standard UNet implementation as a baseline.

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class SimpleBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, dropout_p, transpose_conv, 
                 max_pooling=False, dimensions=2):
        """
        dimensoins = 2 for 2D data, 3 for 3D data.
        """
        super().__init__()
        if dimensions == 2:
            if transpose_conv:
                self.conv = nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride=2, output_padding=1, padding=1)
            else:
                self.conv = nn.Conv2d(in_filters, out_filters, kernel_size, stride=1, padding=1)
            self.norm = nn.BatchNorm2d(out_filters)
            self.dropout = nn.Dropout2d(dropout_p)
            self.pool = F.max_pool2d
            
        elif dimensions == 3:
            if transpose_conv:
                self.conv = nn.ConvTranspose3d(in_filters, out_filters, kernel_size, stride=2, output_padding=1, padding=1)
            else:
                self.conv = nn.Conv3d(in_filters, out_filters, kernel_size, stride=1, padding=1)
            #self.norm = nn.InstanceNorm3d(out_filters)
            self.norm = nn.BatchNorm3d(out_filters)
            self.dropout = nn.Dropout3d(dropout_p)
            self.pool = F.max_pool3d
        else:
            raise ValueError("dimensions can be only 2 or 3 (for 2D or 3D) (int)")
        
        self.do_max_pool = max_pooling
        
    
    def forward(self, x):
        x = self.conv(x)
        if self.do_max_pool:
            x = self.pool(x, 2, 2)
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        return x  
    
    
class Encoder(nn.Module):
    def __init__(self, channels=(3,64,128,256,512,1024), dropout_p=0.1):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [
                SimpleBlock(channels[i], channels[i+1], kernel_size=3, dropout_p=dropout_p, transpose_conv=False) 
                for i in range(len(channels)-1)
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        outs = []
        for block in self.enc_blocks:
            x = block(x)
            outs.append(x)
            x = self.pool(x)
        return outs
    

class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64), dropout_p=0.1):
        super().__init__()
        self.channels = channels
        self.upscale_tconvs = nn.ModuleList(
            [
                SimpleBlock(channels[i], channels[i+1], kernel_size=3, dropout_p=dropout_p, transpose_conv=True)
                for i in range(len(channels) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                SimpleBlock(channels[i], channels[i+1], kernel_size=3, dropout_p=dropout_p, transpose_conv=False)
                for i in range(len(channels) - 1)
            ]
        )
        self.dropout_p = dropout_p
    
    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upscale_tconvs[i](x)
            enc_out = self.crop(encoder_features[i], x)
            #enc_out = encoder_features[i]
            
            x = torch.cat([x, enc_out], dim=1)
            x = self.dec_blocks[i](x)
            
        return x
            
    def crop(self, enc_out, x):
        """
        enc_out: output of encoder at a particular layer in the stack
        x: shape of the data to be matched (the cropping nesseary due to loss of border pixels in convolution ops)
        """
        _, _, H, W = x.shape
        enc_out = torchvision.transforms.CenterCrop([H, W])(enc_out)
        return enc_out
    
class UNet(nn.Module):
    def __init__(self, 
                encoder_channels=(3,64,128,256,512,1024),
                decoder_channels=(1024,512,256,128,64),
                 dropout_p=0.1,
                num_classes=1,
                retain_dim=False,
                output_size=(572,572)
                ):
        super().__init__()
        self.encoder = Encoder(encoder_channels, dropout_p)
        self.decoder = Decoder(decoder_channels, dropout_p)
        self.output_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.retain_dim  = retain_dim
        self.output_size = output_size
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        encoder_features = self.encoder(x)
        # use the [::-1] to reverse the output of the encoder (think the first encoder output goes to the last decoder
        # concatenation etc.
        decoder_out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.output_head(decoder_out)
        
        # match the actual desired output image shape
        if self.retain_dim:
            out = F.interpolate(out, self.output_size)
            
        out = self.soft(out)
        
        return out