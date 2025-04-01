import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchvision.transforms.functional as transforms

def get_conv_func(dims, transpose=False):
    # determine convolution func
        if dims == 2:
            if transpose:
                return nn.ConvTranspose2d
            else:
                return nn.Conv2d
        elif dims == 3:
            if transpose:
                return nn.ConvTranspose3d
            else:
                return nn.Conv3d
        else:
            raise ValueError(f"values of dims of 2 or 3 (2D or 2D conv) are supported only, not {dims}")

class Block(nn.Module):
    def __init__(self, ins, outs, kernel_size, padding='same', transpose_conv=False, dims=2, downsample=False):
        super().__init__()
        # define funcs to build block
        conv_func = get_conv_func(dims, transpose_conv)
        if dims == 2:
            norm_func = nn.InstanceNorm2d
            self.downs = lambda i : F.max_pool2d(i, kernel_size=2)
        else:
            norm_func = nn.InstanceNorm3d
            self.downs = lambda i : F.max_pool3d(i, kernel_size=2)

        self.activ = nn.ReLU() 
        self.conv1 = conv_func(ins, outs, kernel_size=kernel_size, padding=padding)
        self.norm1 = norm_func(outs)
        self.conv2 = conv_func(outs, outs, kernel_size=kernel_size, padding=padding)
        self.norm2 = norm_func(outs)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.norm1(out)

        out = self.activ(out)
        
        if self.downsample:
            out = self.downs(out)


        return out
    
class UpBlock(nn.Module):
    def __init__(self, ins, outs, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(ins, outs, 3, padding=padding)
        self.conv2 = nn.Conv2d(outs, outs, 3, padding=padding)
        self.conv3 = nn.ConvTranspose2d(outs, outs, 2, stride=2)
        self.activ = nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(outs)
        self.norm2 = nn.InstanceNorm2d(outs)
        self.norm3 = nn.InstanceNorm3d(outs)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activ(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activ(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.activ(out)
        
        return out
    
class SysuModel(nn.Module):
    def __init__(self, encoder_sizes=[64, 96, 128, 256, 512], in_channels=3, out_channels=2):
        super().__init__()
        s = len(encoder_sizes) - 1
        self.encoder_blocks = nn.ModuleList(
            [Block(in_channels, encoder_sizes[0], 5, downsample=True)] + [Block(encoder_sizes[i], encoder_sizes[i+1], 3, downsample=True) for i in range(0, s-1)]
        )
        # print(len(self.encoder_blocks))
        
        self.decoder_blocks = nn.ModuleList(
            [UpBlock(encoder_sizes[-2], encoder_sizes[-1])]
            + [UpBlock(encoder_sizes[s-i] + encoder_sizes[s-i-1], encoder_sizes[s-i-1]) for i in range(0, s-1)]
        )
        # print([(encoder_sizes[s-i] + encoder_sizes[s-i-1], encoder_sizes[s-i-1]) for i in range(0, s-1)])
        
        self.end_block = Block(encoder_sizes[0] + encoder_sizes[1], encoder_sizes[0], 3)
        # final layer takes in upsampled versions of every decoder block output (bar the first one)
        # print(torch.Tensor(encoder_sizes)[:-1].sum().item())
        self.final = nn.Conv2d(int(torch.Tensor(encoder_sizes)[:-1].sum().item()), out_channels, 1)
        
    def forward(self, x):
        input_dim = x.shape[-2:]
        
        skips = []
        out = x
        for block in self.encoder_blocks:
            # print(out.shape)
            out = block(out)
            skips.append(out)
            
        decoder_multiscales = []
        
        for i, block in enumerate(self.decoder_blocks):
            # print(out.shape)
            out = block(out)
            if i > 0:
                decoder_multiscales.append(out)
            
            # combine skip layers together 
            sk = skips.pop()
            sk = transforms.center_crop(sk, out.shape[-2:])
            out = torch.cat([out, sk], dim=1)
        
        out = self.end_block(out)
        
        out_dim = out.shape[-2:]
        padx = int((input_dim[0] - out_dim[0]) / 2)
        pady = int((input_dim[1] - out_dim[1]) / 2)
        
        #print(out.shape)
        
        out = F.pad(out, (padx, padx, pady, pady))
        
        # interpolate every item in the list up to the output size
        decoder_multiscales = [F.interpolate(o, size=input_dim, mode='bilinear') for o in decoder_multiscales]
        
        # concatenate the decoder outputs together
        out = torch.cat([out] + decoder_multiscales, dim=1)
        
        out = self.final(out)
        
        return out
        