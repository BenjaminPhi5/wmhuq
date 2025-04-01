import torch
import torch.nn as nn
from trustworthai.models.uq_models.drop_UNet import normalization_layer
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm


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
            
def get_dropout_func(dims):
    if dims == 2:
        return nn.Dropout2d
    if dims == 3:
        return nn.Dropout3d
    else:
        return nn.Dropout


class HM3Block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 dims=2, # 2 =2D, 3=3D,
                 kernel_size=3,
                 dropout_p=0.1,
                 norm_type="bn", # batch norm, or instance 'in' or group 'gn'
                 res_block=True,
                 dropout_dims=None,
                 dropout_both_layers=False,
                 n_power_iterations=1, 
                 eps=1e-12
                ):
        super().__init__()
        
        # determine convolution func
        conv_f = get_conv_func(dims, transpose=False)
        
        # spectral norm wrapper
        spectral_norm_wrapper = lambda layer : spectral_norm(layer, n_power_iterations=n_power_iterations, eps=eps)
            
        if dropout_p > 0:
            dropout_f = get_dropout_func(dims if dropout_dims == None else dropout_dims)
            self.do_dropout = True
        else:
            self.do_dropout = False
        
        # layers needed for the forward pass
        self.conv1 = spectral_norm_wrapper(conv_f(in_channels, out_channels, kernel_size, padding=2, bias=False, dilation=2))
        

        self.dropout1 = None
        if self.do_dropout:
            self.dropout1 = dropout_f()

        self.norm1 = normalization_layer(in_channels, norm=norm_type, dims=dims)()

        self.conv2 = spectral_norm_wrapper(conv_f(out_channels, out_channels, kernel_size, padding=2, bias=False, dilation=2))

        self.dropout2 = None
        if self.do_dropout and dropout_both_layers:
            self.dropout2 = dropout_f()

        self.norm2 = normalization_layer(out_channels, norm=norm_type, dims=dims)()


        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.res_block = res_block
    
    def forward(self, x):
        # print()
        # print("Res UQ Block")
        # print("in shape: ", x.shape)
        out = x
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv1(out)
        # print("conv 1 out shape: ", out.shape)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.dropout2:
            out = self.dropout2(out)
        # print("conv 2 out shape: ", out.shape)
        
        if self.res_block:
            out = torch.add(out, x)
        # print("res out shape: ", out.shape)
        # print("================================")
        return out
    
    def set_applyfunc(self, a):
        for l in self.uq_layers:
            l.set_applyfunc(a)
            
            
class HMFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dims, res_block=False, n_power_iterations=1, eps=1e-12):
        super().__init__()
        
        conv_func = get_conv_func(dims, transpose=False)
        norm_func = normalization_layer(out_channels, norm='in', dims=dims)
        
        # spectral norm wrapper
        spectral_norm_wrapper = lambda layer : spectral_norm(layer, n_power_iterations=n_power_iterations, eps=eps)
        
        self.conv1 = spectral_norm_wrapper(conv_func(in_channels, out_channels, kernel_size=3, dilation=2, padding=2))
        self.norm = norm_func()
        self.lrelu = nn.LeakyReLU(0.01)
        self.conv2 = spectral_norm_wrapper(conv_func(out_channels, out_channels, kernel_size=1))
        self.res_block = res_block
        if self.res_block:
            self.res_connector = conv_func(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # print()
        # print("Feature Block")
        # print("in shape: ", x.shape)
        out = self.conv1(x)
        # print("conv 1 out shape: ", x.shape)
        out = self.norm(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        # print("conv 2 out shape: ", x.shape)
        # print("================================")
        
        if self.res_block:
            out = out + self.res_connector(x)
        
        return out
        
        
class HMUpsampleBlock(nn.Module):
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
    
    
class HyperMapp3rDDU(nn.Module):
    def __init__(self, dims=3,
                 in_channels=3,
                 out_channels=1,
                 encoder_features=[16, 32, 64, 128, 256],
                 decoder_features=[128, 64, 32, 16],
                 softmax=True,
                 up_res_blocks=False,
                 n_power_iterations=3, 
                 eps=1e-12,
                 block_params={
                     "dropout_p":0.1,
                     "norm_type":"in", 
                     "dropout_both_layers":False,
                 }):
        super().__init__()
        
        block_params['n_power_iterations']=n_power_iterations
        block_params['eps']=eps
        
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
            HMFeatureBlock(ins, outs, dims, up_res_blocks, n_power_iterations, eps)
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
        