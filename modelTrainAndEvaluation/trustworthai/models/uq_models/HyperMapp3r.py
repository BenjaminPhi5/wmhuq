import torch
import torch.nn as nn
from trustworthai.models.uq_models.drop_UNet import normalization_layer
import torch.nn.functional as F
from trustworthai.models.uq_models.uq_layers.dropoutconnect import (
    UQDropout,
    UQDropout2d,
    UQDropout3d,
    UQGaussianDropout,
    UQGaussianDropout2d,
    UQGaussianDropout3d,
    UQDropConnect,
    UQDropConnect2d,
    UQDropConnect3d,
    UQGaussianConnect,
    UQGaussianConnect2d,
    UQGaussianConnect3d,
)
from trustworthai.models.uq_models.uq_model import UQModel


# custom block for selecting dropout/drop connect methods and normalization methods
# based on the block I used in my custom resnet
# this is my own custom implementation from scratch, not entirely sure if it is corrects

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


class HM3Block(UQModel):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 dims=2, # 2 =2D, 3=3D,
                 kernel_size=3,
                 dropout_type="bernoulli",
                 dropout_p=0.1,
                 gaussout_mean=1, # NOTE THE PREDICT STEP CURRENTLY ONLY SUPPORTS MEAN = 1
                 dropconnect_type="bernoulli",
                 dropconnect_p=0.1,
                 gaussconnect_mean=1,
                 norm_type="bn", # batch norm, or instance 'in' or group 'gn'
                 use_multidim_dropout = True, # use 2d or 3d dropout instead of 1d dropout. applies to gaussian dropout too
                 use_multidim_dropconnect = True, # use 2d or 3d dropconnect instead of 1d dropconnect, applies to gaussian dropconnect too
                 groups=1,
                 gn_groups=4, # number of groups for group norm normalization.
                 uq_layer_on_conv2=False,
                 res_block=True
                ):
        super().__init__()
        
        # determine convolution func
        conv_f = get_conv_func(dims, transpose=False)
            
        # determine dropout func
        if dropout_type:
            # standard dropout
            if dropout_type == "bernoulli":
                if use_multidim_dropout:
                    if dims == 2:
                        dropout_f = UQDropout2d
                    else:
                        dropout_f = UQDropout3d
                else:
                    dropout_f = UQDropout
                    
            # gaussian dropout    
            elif dropout_type == "gaussian":
                if use_multidim_dropout:
                    if dims == 2:
                        dropout_f = UQGaussianDropout2d
                    elif dims == 3:
                        dropout_f = UQGaussianDropout3d
                else:
                    dropout_f = UQGaussianDropout
            else:
                raise ValueError(f"dropout type {dropout_type} not supported, "
                                 "only 'bernoulli' or 'gaussian' are supported")
        # no dropout
        else:
            dropout_f = None
        
        # determine dropconnect function
        if dropconnect_type:
            # standard dropconnect
            if dropconnect_type == "bernoulli":
                if use_multidim_dropout:
                    if dims == 2:
                        dropconnect_f = UQDropConnect2d
                    else:
                        dropconnect_f = UQDropConnect3d
                else:
                    dropconnect_f = UQDropConnect
                    
            # gaussian dropout    
            elif dropconnect_type == "gaussian":
                if use_multidim_dropconnect:
                    if dims == 2:
                        dropconnect_f = UQGaussianConnect2d
                    elif dims == 3:
                        dropconnect_f = UQGaussianConnect3d
                else:
                    dropconnect_f = UQGaussianConnect
            else:
                raise ValueError(f"dropconnect type {dropconnect_type} not supported, "
                                 "only 'bernoulli' or 'gaussian' are supported")
        else:
            dropconnect_f = None
        
        self.uq_layers = []

        # layers needed for the forward pass
        self.conv1 = conv_f(in_channels, out_channels, kernel_size, padding=2, bias=False, dilation=2)
        if dropconnect_f:
            if dropconnect_type == "bernoulli":
                self.convout1 = dropconnect_f(self.conv1, None, dropconnect_p)
            else:
                self.convout1 = dropconnect_f(self.conv1, None, gaussconnect_mean, dropconnect_p)
            self.uq_layers.append(self.convout1)
        else:
            self.convout1 = self.conv1

        if dropout_f:
            if dropout_type == "bernoulli":
                self.dropout1 = dropout_f(dropout_p)
            else:
                self.dropout1 = dropout_f(gaussout_mean, dropout_p)
            self.uq_layers.append(self.dropout1)
        else:
            self.dropout1 = None

        self.norm1 = normalization_layer(in_channels, norm=norm_type, gn_groups=gn_groups, dims=dims)()

        self.conv2 = conv_f(out_channels, out_channels, kernel_size, padding=2, bias=False, dilation=2)
        if dropconnect_f and uq_layer_on_conv2:
            if dropconnect_type == "bernoulli":
                self.convout2 = dropconnect_f(self.conv2, None, dropconnect_p)
            else:
                self.convout2 = dropconnect_f(self.conv2, None, gaussconnect_mean, dropconnect_p)
            self.uq_layers.append(self.convout2)
        else:
            self.convout2 = self.conv2

        if dropout_f and uq_layer_on_conv2:
            if dropout_type == "bernoulli":
                self.dropout2 = dropout_f(dropout_p)
            else:
                self.dropout2 = dropout_f(gaussout_mean, dropout_p)
            self.uq_layers.append(self.dropout1)
        else:
            self.dropout2 = None

        self.norm2 = normalization_layer(out_channels, norm=norm_type, gn_groups=gn_groups, dims=dims)()


        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.res_block = res_block
    
    def forward(self, x):
        # print()
        # print("Res UQ Block")
        # print("in shape: ", x.shape)
        out = x
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.convout1(out)
        # print("conv 1 out shape: ", out.shape)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        out = self.convout2(out)
        # print("conv 2 out shape: ", out.shape)
        
        if self.res_block:
            out = torch.add(out, x)
        # print("res out shape: ", out.shape)
        # print("================================")
        return out
    
    def set_applyfunc(self, a):
        for l in self.uq_layers:
            l.set_applyfunc(a)
            
            
class HMFeatureBlock(UQModel):
    def __init__(self, in_channels, out_channels, dims):
        super().__init__()
        
        conv_func = get_conv_func(dims, transpose=False)
        norm_func = normalization_layer(out_channels, norm='in', dims=dims)
        
        self.conv1 = conv_func(in_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.norm = norm_func()
        self.lrelu = nn.LeakyReLU(0.01)
        self.conv2 = conv_func(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # print()
        # print("Feature Block")
        # print("in shape: ", x.shape)
        x = self.conv1(x)
        # print("conv 1 out shape: ", x.shape)
        x = self.norm(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        # print("conv 2 out shape: ", x.shape)
        # print("================================")
        return x
        
        
class HMUpsampleBlock(UQModel):
    def __init__(self, in_channels, out_channels, dims):
        super().__init__()
        
        # determine convolution func
        conv_func = get_conv_func(dims, transpose=True)
        
        self.norm1 = normalization_layer(in_channels, norm='in', dims=dims)()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)
        self.up_conv = conv_func(in_channels, out_channels, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.norm2 = normalization_layer(out_channels, norm='in', dims=dims)()
        
    def forward(self, x):
        # print()
        # print("Upsample Block")
        # print("in shape: ", x.shape)
        x = self.norm1(x)
        x = self.lrelu(x)
        x = self.up_conv(x)
        # print("conv 1 out shape: ", x.shape)
        x = self.norm2(x)
        x = self.lrelu(x)
        
        # print("================================")
        return x
    
    
class HyperMapp3r(UQModel):
    def __init__(self, dims=3,
                 in_channels=3,
                 out_channels=1,
                 encoder_features=[16, 32, 64, 128, 256],
                 decoder_features=[128, 64, 32, 16],
                 softmax=True,
                 block_params={
                     "dropout_type":"bernoulli",
                     "dropout_p":0.1,
                     "gaussout_mean":None, 
                     "dropconnect_type":None,
                     "dropconnect_p":None,
                     "gaussconnect_mean":None,
                     "norm_type":"in", 
                     "use_multidim_dropout":True, 
                     "use_multidim_dropconnect":True, 
                     "uq_layer_on_conv2":False,
                 }):
        super().__init__()
        
        # print("dims: ", dims)
        
        conv_func = get_conv_func(dims, transpose=False)
        # print("conv func: ", conv_func)
        
        self.encoder_resuq_blocks = nn.ModuleList([
            HM3Block(fs, fs, dims, **block_params)
            for fs in encoder_features
        ])
        self.encoder_down_blocks = nn.ModuleList([
            conv_func(ins, outs, kernel_size=3, stride=2, padding=1)
            for (ins, outs) in zip([in_channels] + encoder_features[:-1], encoder_features)
        ])
        
        self.decoder_feature_blocks = nn.ModuleList([
            HMFeatureBlock(ins, outs, dims)
            for (ins, outs) in zip([f * 2 for f in decoder_features[:-1]], decoder_features[:-1])
        ])
        
        self.decoder_upsample_blocks = nn.ModuleList([
            HMUpsampleBlock(ins, outs, dims)
            for (ins, outs) in zip([f * 2 for f in decoder_features], decoder_features)
        ])
        
        
        self.skip_final_convs = nn.ModuleList([
            conv_func(fs, out_channels, kernel_size=1)
            for fs in decoder_features[1:-1]
        ])
        
        final_a_features = encoder_features[0] * 2
        # print("final a features: ", final_a_features)
        self.final_a = conv_func(final_a_features, final_a_features, kernel_size=3, stride=1, padding=1)
        # print("final a weight size: ", self.final_a.weight.shape)
        self.final_b = conv_func(final_a_features, out_channels, kernel_size=1)
        
        self.lrelu = nn.LeakyReLU(0.01)
        mode = "bilinear" if dims == 2 else "trilinear"
        self.interpolate = lambda x : F.interpolate(x, scale_factor=2, mode=mode)
        self.softmax = nn.Softmax(dim=1) if softmax else None
        
        
        self.down_steps = len(self.encoder_down_blocks)
        self.up_steps = len(self.decoder_upsample_blocks)
        
        
    def forward(self, x):
        skip_conns = []
        out = x
        
        # print("hypermappr3")
        # print("in shape: ", x.shape)
        # print("~~ENCODER~~")
        # encoder path
        for l in range(self.down_steps):
            out = self.encoder_down_blocks[l](out)
            out = self.encoder_resuq_blocks[l](out)
            # print("encoder group out shape", out.shape)
            
            if l != self.down_steps-1:
                skip_conns.append(out)
                
        # decoder path
        # print("~~DECODER~~")
        out = self.decoder_upsample_blocks[0](out)
        secondary_skip_conns = []
        for l in range(1, self.up_steps):
            # print("decoder group in: ", out.shape)
            #print("skip conn shape: ", skip_conns[-1].shape)
            out = torch.cat([out, skip_conns.pop()], dim=1)
            #print("post cat shape: ", out.shape)
            out = self.decoder_feature_blocks[l-1](out)
            out = self.decoder_upsample_blocks[l](out)
            
            if l >= 1:
                secondary_skip_conns.append(out)
        
        #print("final cat in shape: ", out.shape)
        out = torch.cat([out, skip_conns.pop()], dim=1)
        #print("post cat shape: ", out.shape)
        out = self.final_a(out)
        out = self.lrelu(out)
        out = self.final_b(out)
        #print("main branch otu shape: ", out.shape)
        
        # combine secondary skips
        sk1 = self.skip_final_convs[0](secondary_skip_conns[0])
        #print("sk1 out shape pre interpolate: ", sk1.shape)
        sk1 = self.interpolate(sk1)
        #print("sk1 out shape post interpolate: ", sk1.shape)
        sk2 = self.skip_final_convs[1](secondary_skip_conns[1])
        #print("sk2 out shape pre interpolate: ", sk2.shape)
        sk2 = torch.add(sk1, sk2)
        #print("sk2 out shape post add: ", sk2.shape)
        sk2 = self.interpolate(sk2)
        #print("sk2 out shape post interpolate: ", sk2.shape)
        
        out = torch.add(out, sk2)
        
        out = self.interpolate(out)
        
        if self.softmax:
            out = self.softmax(out)
        
        return out
        
        
        
"""

- what is the kernel size for their deconv block? ive put three
- what is their l_relu parameter? I have put 0.01 (todo make as a gloabl const)
- what do they do about the output shape, do they upsample or no its strange
- I think its not great the way they do the upsampling at the last layer, would be better
- to have a neural net layer do the upscale I think...
- need to try and use the kernel sizes given in the paper as well (they have a few 7x7 ones...
"""
        