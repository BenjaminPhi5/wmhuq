import torch
import torch.nn as nn
from trustworthai.models.uq_models.uq_model import UQModel
from trustworthai.models.uq_models.uq_layers.uq_generic_layer import UQLayerWrapper
 
# various dropout and dropconnect layers
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

def normalization_layer(planes, norm='gn', gn_groups=None, dims=2, as_uq_layer=False):
    if as_uq_layer:
        wrapper = lambda l : UQLayerWrapper(l)
    else:
        wrapper = lambda x : x
    if dims == 2:
        if norm == "bn":
            return lambda : wrapper(nn.BatchNorm2d(planes))
        elif norm == "gn":
            return lambda : wrapper(nn.GroupNorm(gn_groups, planes)) # it does 2d auomatically?
        elif norm == "in":
            return lambda : wrapper(nn.InstanceNorm2d(planes))
        else:
            raise ValueError(f"norm type {norm} not supported, only 'bn', 'in', or 'gn' supported")
    elif dims == 3:
        if norm == "bn":
            return lambda : wrapper(nn.BatchNorm3d(planes))
        elif norm == "gn":
            return lambda : wrapper(nn.GroupNorm(gn_groups, planes)) # it does 3d automatically?
        elif norm == "in":
            return lambda : wrapper(nn.InstanceNorm3d(planes))
        else:
            raise ValueError(f"norm type {norm} not supported, only 'bn', 'in', or 'gn' supported")

# custom block for selecting dropout/drop connect methods and normalization methods
class Block(UQModel):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 name,
                 dims=2, # 2 =2D, 3=3D,
                 kernel_size=3,
                 dropout_type="bernoulli",
                 dropout_p=0.1,
                 gaussout_mean=1, # NOTE THE PREDICT STEP CURRENTLY ONLY SUPPORTS MEAN = 1
                 dropconnect_type="bernoulli",
                 dropconnect_p=0.1,
                 gaussconnect_mean=1,
                 norm_type="bn", # batch norm, or instance 'in' or group 'gn'
                 use_uq_norm_layer=False,
                 use_multidim_dropout=True, # use 2d or 3d dropout instead of 1d dropout. applies to gaussian dropout too
                 use_multidim_dropconnect = True, # use 2d or 3d dropconnect instead of 1d dropconnect, applies to gaussian dropconnect too
                 groups=1,
                 gn_groups=4, # number of groups for group norm normalization.
                ):
        super().__init__()
        
        # determine convolution func
        if dims == 2:
            conv_f = nn.Conv2d
        elif dims == 3:
            conv_f = nn.Conv3d
        else:
            raise ValueError(f"values of dims of 2 or 3 (2D or 2D conv) are supported only, not {dims}")
            
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
    
        # determine normalization type
        norm_layer = normalization_layer(out_channels, norm=norm_type, gn_groups=gn_groups, dims=dims, as_uq_layer=use_uq_norm_layer)

        # layers needed for the forward pass
        self.conv1 = conv_f(in_channels, out_channels, kernel_size, padding=1, bias=False)
        if dropconnect_f:
            if dropconnect_type == "bernoulli":
                self.convout1 = dropconnect_f(self.conv1, None, dropconnect_p)
            else:
                self.convout1 = dropconnect_f(self.conv1, None, gaussconnect_mean, dropconnect_p)
        else:
            self.convout1 = self.conv1

        if dropout_f:
            if dropout_type == "bernoulli":
                self.dropout1 = dropout_f(dropout_p)
            else:
                self.dropout1 = dropout_f(gaussout_mean, dropout_p)
        else:
            self.dropout1 = None

        self.norm1 = norm_layer()

        self.conv2 = conv_f(out_channels, out_channels, kernel_size, padding=1, bias=False)
        if dropconnect_f:
            if dropconnect_type == "bernoulli":
                self.convout2 = dropconnect_f(self.conv2, None, dropconnect_p)
            else:
                self.convout2 = dropconnect_f(self.conv2, None, gaussconnect_mean, dropconnect_p)
        else:
            self.convout2 = self.conv2

        if dropout_f:
            if dropout_type == "bernoulli":
                self.dropout2 = dropout_f(dropout_p)
            else:
                self.dropout2 = dropout_f(gaussout_mean, dropout_p)
        else:
            self.dropout2 = None

        self.norm2 = norm_layer()


        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.convout1(x)
        x = self.norm1(x)
        if self.dropout1:
            x = self.dropout1(x)
        x = self.relu(x)
        
        x = self.convout2(x)
        x = self.norm2(x)
        if self.dropout2:
            x = self.dropout2(x)
        x = self.relu(x)
        
        return x

class UNet(UQModel):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, softmax=True,
                 kernel_size=3,
                 dropout_type="bernoulli",
                 dropout_p=0.1,
                 gaussout_mean=1, # NOTE THE PREDICT STEP CURRENTLY ONLY SUPPORTS MEAN = 1
                 dropconnect_type="bernoulli",
                 dropconnect_p=0.1,
                 gaussconnect_mean=1,
                 norm_type="bn", # batch norm, or instance 'in' or group 'gn'
                 use_uq_norm_layer=False,
                 use_multidim_dropout = True, # use 2d or 3d dropout instead of 1d dropout. applies to gaussian dropout too
                 use_multidim_dropconnect = True, # use 2d or 3d dropconnect instead of 1d dropconnect, applies to gaussian dropconnect too
                 groups=1,
                 gn_groups=4, # number of groups for group norm normalization.
                ):
        super().__init__()
                 
        block_params = {"dims":2, "kernel_size":kernel_size,"dropout_type":dropout_type,
                        "dropout_p":dropout_p,"gaussout_mean":gaussout_mean,
                        "dropconnect_p":dropconnect_p,"dropconnect_type":dropconnect_type,"gaussconnect_mean":gaussconnect_mean,
                        "norm_type":norm_type,"use_uq_norm_layer":use_uq_norm_layer,"use_multidim_dropout":use_multidim_dropout,
                        "use_multidim_dropconnect":use_multidim_dropconnect,"groups":groups,
                        "gn_groups":gn_groups,
                       }

        features = init_features
        self.encoder1 = Block(in_channels, features, name="enc1", **block_params)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Block(features, features * 2, name="enc2",**block_params)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Block(features * 2, features * 4, name="enc3", **block_params)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Block(features * 4, features * 8, name="enc4", **block_params)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Block(features * 8, features * 16, name="bottleneck", **block_params)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Block((features * 8) * 2, features * 8, name="dec4", **block_params)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Block((features * 4) * 2, features * 4, name="dec3", **block_params)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Block((features * 2) * 2, features * 2, name="dec2", **block_params)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features, name="dec1", **block_params)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.do_softmax = softmax

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)
        if self.do_softmax:
            return torch.nn.functional.softmax(out, dim=1)
        else:
            return out