# adapted from https://github.com/jeya-maria-jose/KiU-Net-pytorch

# TODO: complete the 3D version, see the repo they do it slightly different again

import torch.nn as nn
import torch.nn.functional as F
import torch


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
    
    
def block_list(in_lists, out_lists, kernel_size=3, dropout_p=0.5, do_transpose_conv=False, max_pooling=True, dimensions=2):
    """
    return a module containing a list of blocks, (not connected in any way, but will be 
    properly recognised as model weights)
    """
    return nn.ModuleList([
        SimpleBlock(ins, outs, kernel_size, dropout_p, do_transpose_conv, max_pooling, dimensions)
        for (ins, outs) in
        zip(in_lists, out_lists)
    ])
    

class FullyConvolutionalAutoEncoder(nn.Module):
    def __init__(self, encoder_layers=[64, 128, 256, 512], decoder_layers=[512, 256, 128, 64], in_channels=3, out_channels=1, dimensions=2):
        super().__init__()
        
        self.encoder_blocks = block_list([in_channels] + encoder_layers[:-1], encoder_layers,
                                        3, 0.5, False, max_pooling=True, dimensions=dimensions)
        
        self.decoder_blocks = block_list([encoder_layers[-1]] + decoder_layers, decoder_layers + [out_channels],
                                        3, 0.5, True, max_pooling=False, dimensions=dimensions)
        
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x, encode=False):
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
        
        if encode:
            return x
            
        for dec_block in self.decoder_blocks:
            x = dec_block(x)
            
        x = self.soft(x)
        return x
    
    
class KiUNetWithTranspose(nn.Module):
    # I have slightly adapted this to use transpose convolution layers as opposed to interpolation...
    # not sure if that is the best idea or not
    # the standard version (class below, uses interpolation like they do)
    def __init__(self, encoder_layers=[16,32,64], decoder_layers=[32,16,8], 
                 encoderf1_layers=[16,32,64], decoderf1_layers=[32,16,8], 
                 intere_layers=[16,32,64], interd_layers=[32,16],
                         in_channels=3, out_channels=1, kernel_size=3, dropout_p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        dimensions=2
        
        
        self.encoder_blocks = block_list([in_channels] + encoder_layers[:-1], encoder_layers,
                                        kernel_size, dropout_p, False, max_pooling=True, dimensions=dimensions)
        
        self.decoder_blocks = block_list([encoder_layers[-1]] + decoder_layers[:-1], decoder_layers,
                                        kernel_size, dropout_p, True, max_pooling=False, dimensions=dimensions)
        
        self.encoderf1_blocks = block_list([in_channels] + encoderf1_layers[:-1], encoderf1_layers,
                                          kernel_size, dropout_p, True, max_pooling=False, dimensions=dimensions)
        
        self.decoderf1_blocks = block_list([encoderf1_layers[-1]] + decoderf1_layers[:-1], decoderf1_layers,
                                          kernel_size, dropout_p, False, max_pooling=True, dimensions=dimensions)
        
        self.intere1_blocks = block_list(intere_layers, intere_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.intere2_blocks = block_list(intere_layers, intere_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.interd1_blocks = block_list(interd_layers, interd_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.interd2_blocks = block_list(interd_layers, interd_layers,
                                        kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.final_conv = nn.Conv2d(8, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)
        self.mode = "bilinear"
        
    def forward(self, x):
        
        unet_out = x
        kinet_out = x
        skip_conns_stack = []
        scale_factor = 1
        
        # encoder path
        num_blocks = len(self.encoder_blocks)
        for l in range(num_blocks):
            # standard forward pass for both paths
            unet_out = self.encoder_blocks[l](unet_out)
            kinet_out = self.encoderf1_blocks[l](kinet_out)
            
            tmp = unet_out
            
            # CRFB block
            scale_factor *= 4.
            crfb1 = self.intere1_blocks[l](kinet_out)
            unet_out = torch.add(
                unet_out,
                F.interpolate(crfb1, scale_factor=1./scale_factor, mode=self.mode)
            )
            
            crfb2 = self.intere2_blocks[l](tmp)
            kinet_out = torch.add(
                kinet_out,
                F.interpolate(crfb2, scale_factor=scale_factor,mode=self.mode)
            )
            
            # append skip connections
            if l != num_blocks - 1:
                skip_conns_stack.append((unet_out, kinet_out))
            
            
        # decoder path
        for l in range(len(self.decoder_blocks)):
            # standard forward pass for both paths
            unet_out = self.decoder_blocks[l](unet_out)
            kinet_out = self.decoderf1_blocks[l](kinet_out)
            
            tmp = unet_out
            
            # CRFB block
            if l != num_blocks - 1:
                scale_factor /= 4.
                crfb1 = self.interd1_blocks[l](kinet_out)
                unet_out = torch.add(
                    unet_out,
                    F.interpolate(crfb1, scale_factor=1./scale_factor, mode=self.mode)
                )

                crfb2 =self.interd2_blocks[l](tmp)
                kinet_out = torch.add(
                    kinet_out,
                    F.interpolate(crfb2, scale_factor=scale_factor, mode=self.mode)
                )
            
            # add skip connections
            if l != num_blocks - 1:
                unet_skipc, kinet_skipc = skip_conns_stack.pop() # pop implements LIFO stack behaviour
                unet_out = torch.add(unet_out, unet_skipc)
                kinet_out = torch.add(kinet_out, kinet_skipc)
            
        
        # fusion of both branches
        out = torch.add(unet_out, kinet_out)
        out = self.final_conv(out)
        out = self.soft(out)
        
        return out
    
    
class KiUNet(nn.Module):
    # I have slightly adapted this to use transpose convolution layers as opposed to interpolation...
    # not sure if that is the best idea or not
    # the standard version (class below, uses interpolation like they do)
    def __init__(self, encoder_layers=[16,32,64], decoder_layers=[32,16,8], 
                 encoderf1_layers=[16,32,64], decoderf1_layers=[32,16,8], 
                 intere_layers=[16,32,64], interd_layers=[32,16],
                 in_channels=3, out_channels=1, kernel_size=3, dropout_p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        dimensions=2
        
        
        self.encoder_blocks = block_list([in_channels] + encoder_layers[:-1], encoder_layers,
                                        kernel_size, dropout_p, False, max_pooling=True, dimensions=dimensions)
        
        self.decoder_blocks = block_list([encoder_layers[-1]] + decoder_layers[:-1], decoder_layers,
                                        kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.encoderf1_blocks = block_list([in_channels] + encoderf1_layers[:-1], encoderf1_layers,
                                          kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.decoderf1_blocks = block_list([encoderf1_layers[-1]] + decoderf1_layers[:-1], decoderf1_layers,
                                          kernel_size, dropout_p, False, max_pooling=True, dimensions=dimensions)
        
        self.intere1_blocks = block_list(intere_layers, intere_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.intere2_blocks = block_list(intere_layers, intere_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.interd1_blocks = block_list(interd_layers, interd_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.interd2_blocks = block_list(interd_layers, interd_layers,
                                        kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.final_conv = nn.Conv2d(8, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)
        
        self.mode = 'bilinear'
        self.upscale = lambda x : F.interpolate(x, scale_factor=2, mode=self.mode)
        
    def forward(self, x):
        
        unet_out = x
        kinet_out = x
        skip_conns_stack = []
        scale_factor = 1
        
        # encoder path
        num_blocks = len(self.encoder_blocks)
        for l in range(num_blocks):
            # standard forward pass for both paths
            unet_out = self.encoder_blocks[l](unet_out)
            kinet_out = self.upscale(self.encoderf1_blocks[l](kinet_out))
            
            tmp = unet_out
            
            # CRFB block
            scale_factor *= 4.
            crfb1 = self.intere1_blocks[l](kinet_out)
            unet_out = torch.add(
                unet_out,
                F.interpolate(crfb1, scale_factor=1./scale_factor, mode=self.mode)
            )
            
            crfb2 = self.intere2_blocks[l](tmp)
            kinet_out = torch.add(
                kinet_out,
                F.interpolate(crfb2, scale_factor=scale_factor,mode=self.mode)
            )
            
            # append skip connections
            if l != num_blocks - 1:
                skip_conns_stack.append((unet_out, kinet_out))
            
            
        # decoder path
        for l in range(len(self.decoder_blocks)):
            # standard forward pass for both paths
            unet_out = self.upscale(self.decoder_blocks[l](unet_out))
            kinet_out = self.decoderf1_blocks[l](kinet_out)
            
            tmp = unet_out
            
            # CRFB block
            if l != num_blocks - 1:
                scale_factor /= 4.
                crfb1 = self.interd1_blocks[l](kinet_out)
                unet_out = torch.add(
                    unet_out,
                    F.interpolate(crfb1, scale_factor=1./scale_factor, mode=self.mode)
                )

                crfb2 =self.interd2_blocks[l](tmp)
                kinet_out = torch.add(
                    kinet_out,
                    F.interpolate(crfb2, scale_factor=scale_factor, mode=self.mode)
                )
            
            # add skip connections
            if l != num_blocks - 1:
                unet_skipc, kinet_skipc = skip_conns_stack.pop() # pop implements LIFO stack behaviour
                unet_out = torch.add(unet_out, unet_skipc)
                kinet_out = torch.add(kinet_out, kinet_skipc)
            
        
        # fusion of both branches
        out = torch.add(unet_out, kinet_out)
        out = self.final_conv(out)
        out = self.soft(out)
        
        return out
    
class KiUNet3D(nn.Module):
    # I have slightly adapted this to use transpose convolution layers as opposed to interpolation...
    # not sure if that is the best idea or not
    # the standard version (class below, uses interpolation like they do)
    def __init__(self, encoder_layers=[16,32,64], decoder_layers=[32,16,8], 
                 encoderf1_layers=[16,32,64], decoderf1_layers=[32,16,8], 
                 intere_layers=[16,32,64], interd_layers=[32,16],
                 in_channels=3, out_channels=1, kernel_size=3, dropout_p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        dimensions = 3
        
        
        self.encoder_blocks = block_list([in_channels] + encoder_layers[:-1], encoder_layers,
                                        kernel_size, dropout_p, False, max_pooling=True, dimensions=dimensions)
        
        self.decoder_blocks = block_list([encoder_layers[-1]] + decoder_layers[:-1], decoder_layers,
                                        kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.encoderf1_blocks = block_list([in_channels] + encoderf1_layers[:-1], encoderf1_layers,
                                          kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.decoderf1_blocks = block_list([encoderf1_layers[-1]] + decoderf1_layers[:-1], decoderf1_layers,
                                          kernel_size, dropout_p, False, max_pooling=True, dimensions=dimensions)
        
        self.intere1_blocks = block_list(intere_layers, intere_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.intere2_blocks = block_list(intere_layers, intere_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.interd1_blocks = block_list(interd_layers, interd_layers, 
                                         kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.interd2_blocks = block_list(interd_layers, interd_layers,
                                        kernel_size, dropout_p, False, max_pooling=False, dimensions=dimensions)
        
        self.final_conv = nn.Conv3d(8, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)
        
        mode = 'trilinear'
        self.upscale_ki = lambda x : F.interpolate(x, scale_factor=(1,2,2), mode=mode)
        self.upscale_u = lambda x : F.interpolate(x, scale_factor=2, mode=mode)
        
    def forward(self, x):
        
        unet_out = x
        kinet_out = x
        skip_conns_stack = []
        scale_factor = 1
        mode = 'trilinear'
        
        # encoder path
        num_blocks = len(self.encoder_blocks)
        for l in range(num_blocks):
            # standard forward pass for both paths
            unet_out = self.encoder_blocks[l](unet_out)
            kinet_out = self.upscale_ki(self.encoderf1_blocks[l](kinet_out))
            
            # print("unet out shape: ", unet_out.shape)
            # print("kinet out shape: ", kinet_out.shape)
            
            tmp = unet_out
            
            # CRFB block
            scale_factor *= 4.
            crfb1 = self.intere1_blocks[l](kinet_out)
            # print("crbf unet out shape: ", crfb1.shape)
            unet_out = torch.add(
                unet_out,
                F.interpolate(crfb1, size=unet_out.size()[2:], scale_factor=None, mode=mode)
            )
            # print("crbf add unet out shape: ", unet_out.shape)
            
            crfb2 = self.intere2_blocks[l](tmp)
            # print("crbf kinet out shape: ", crfb2.shape)
            # print(kinet_out.size())
            interpolated = F.interpolate(crfb2, size=kinet_out.size()[2:],scale_factor=None,mode=mode)
            # print("interpolated shape: ", interpolated.shape)
            kinet_out = torch.add(
                kinet_out,
                interpolated
            )
            # print("crbf add kinet out shape: ", kinet_out.shape)
            
            # append skip connections
            if l != num_blocks - 1:
                skip_conns_stack.append((unet_out, kinet_out))
            # print()
            
        # print("decoder\n")
            
        # decoder path
        for l in range(len(self.decoder_blocks)):
            # standard forward pass for both paths
            unet_out = self.upscale_u(self.decoder_blocks[l](unet_out))
            kinet_out = self.decoderf1_blocks[l](kinet_out)
            
            # print("unet out shape: ", unet_out.shape)
            # print("kinet out shape: ", kinet_out.shape)
            
            tmp = unet_out
            
            # CRFB block
            if l != num_blocks - 1:
                scale_factor /= 4.
                crfb1 = self.interd1_blocks[l](kinet_out)
                # print("crbf unet out shape: ", crfb1.shape)
                unet_out = torch.add(
                    unet_out,
                    F.interpolate(crfb1, size=unet_out.size()[2:], scale_factor=None, mode=mode)
                )
                # print("crbf add unet out shape: ", unet_out.shape)

                crfb2 =self.interd2_blocks[l](tmp)
                # print("crbf kinet out shape: ", crfb2.shape)
                # print(kinet_out.size())
                interpolated = F.interpolate(crfb2, size=kinet_out.size()[2:], scale_factor=None, mode=mode)
                # print("interpolated shape: ", interpolated.shape)
                kinet_out = torch.add(
                    kinet_out,
                    interpolated
                )
                # print("crbf add kinet out shape: ", kinet_out.shape)
            # print()
            
            # add skip connections
            if l != num_blocks - 1:
                unet_skipc, kinet_skipc = skip_conns_stack.pop() # pop implements LIFO stack behaviour
                # print("skip size unet: ", unet_skipc.shape)
                # print("skip size kinet: ", kinet_skipc.shape)
                # print("unet size ", unet_out.shape)
                # print("kinet size", kinet_out.shape)
                # do padding where the sizes don't match
                if unet_out.shape != unet_skipc.shape:
                    # print("using")
                    unet_out = pad_3D_tensors(unet_out, unet_skipc.shape)
                    # print("new unet shape: ", unet_out.shape)
                if kinet_out.shape != kinet_skipc.shape:
                    kinet_out = pad_3D_tensors(kinet_out, kinet_skipc.shape)
                unet_out = torch.add(unet_out, unet_skipc)
                kinet_out = torch.add(kinet_out, kinet_skipc)
            
        
        # fusion of both branches
        # print("out_shapes: ", kinet_out.shape, unet_out.shape)
        kinet_out = F.interpolate(kinet_out, scale_factor=(2,1,1), mode='trilinear')
        if kinet_out.shape != unet_out.shape:
            unet_out = pad_3D_tensors(unet_out, kinet_out.shape)
        out = torch.add(unet_out, kinet_out)
        out = self.final_conv(out)
        out = self.soft(out)
        
        return out
    

def pad_3D_tensors(img, target_shape):
    img_shape = img.shape
    lr_diff = target_shape[-3] - img_shape[-3]
    top_bottom_diff = target_shape[-2] - img_shape[-2]
    front_back_diff = target_shape[-1] - img_shape[-1]
    l_pad = lr_diff // 2
    r_pad = lr_diff - l_pad
    top_pad = top_bottom_diff // 2
    bottom_pad = top_bottom_diff - top_pad
    front_pad = front_back_diff // 2
    back_pad = front_back_diff - front_pad
    
    #print((l_pad, r_pad, top_pad, bottom_pad, front_pad, back_pad))
    
    return F.pad(img, (front_pad, back_pad, top_pad, bottom_pad, l_pad, r_pad))