import torch
import torch.nn.functional as F
import torch.nn as nn
from twaibrain.braintorch.models.ssn import SSN_Head

class SSN_Wrapped_Deep_Supervision(nn.Module):
    def __init__(self, base_unet, deep_supervision_levels, ssn_config):
        super().__init__()
        # base_unet is your nnU-Net model (e.g., from MONAI) that has multiple decoder outputs
        self.base_unet = base_unet

        # Create an SSN instance for each deep supervision level output
        # ssn_config is a dict containing 'intermediate_channels', 'out_channels', 'dims', etc.
        self.ssn_heads = nn.ModuleList([
            SSN_Head(
                intermediate_channels=ssn_config['intermediate_channels'],
                out_channels=ssn_config['out_channels'],
                dims=ssn_config['dims'],
                rank=ssn_config.get('rank', 25),
                diagonal=ssn_config.get('diagonal', False),
                epsilon=ssn_config.get('epsilon', 1e-5),
                return_cpu_dist=ssn_config.get('return_cpu_dist', False)
            )
            for _ in range(deep_supervision_levels)
        ])

    def forward(self, x):
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)

        # Apply SSN to each decoder output
        ssn_outputs = []
        for ssn_head, decoder_output in zip(self.ssn_heads, outputs):
            mask_l = F.interpolate(mask, size=decoder_output.shape[2:], mode='nearest')
            ssn_outputs.append(ssn_head(decoder_output, mask_l))

        return ssn_outputs

    def mean(self, x):
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)
        
        return self.ssn_heads[0](outputs[0], mask)['logit_mean']

    def _samples_from_dist(self, dist, num_samples, rsample=True, symmetric=True):
        if symmetric:
            assert num_samples % 2 == 0
            num_samples = num_samples // 2

        samples = dist.rsample((num_samples,)) if rsample else dist.sample((num_samples,))
        
        if symmetric:
            mean = dist.mean
            samples = samples - mean
            return torch.cat([samples, -samples]) + mean
        else:
            return samples
    
    def mean_and_sample(self, x, num_samples, rsample=True, symmetric=True):
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)
        out = self.ssn_heads[0](outputs[0], mask)
        mean = out['logit_mean']
        dist = out['distribution']
        samples = self._samples_from_dist(dist, num_samples, rsample, symmetric)
        return mean, samples

class SSN_Wrapped_Deep_Supervision_LLO(nn.Module):
    """
    only does SSN on the last layer, not all the layers
    """
    def __init__(self, base_unet, ssn_config):
        super().__init__()
        # base_unet is your nnU-Net model (e.g., from MONAI) that has multiple decoder outputs
        self.base_unet = base_unet

        # Create an SSN instance for each deep supervision level output
        # ssn_config is a dict containing 'intermediate_channels', 'out_channels', 'dims', etc.
        self.ssn_head = SSN_Head(
            intermediate_channels=ssn_config['intermediate_channels'],
            out_channels=ssn_config['out_channels'],
            dims=ssn_config['dims'],
            rank=ssn_config['rank'],
            diagonal=ssn_config.get('diagonal', False),
            epsilon=ssn_config.get('epsilon', 1e-5),
            return_cpu_dist=ssn_config.get('return_cpu_dist', False)
        )

    def forward(self, x):
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)

        # Apply SSN to the last output
        outputs[0] = self.ssn_head(outputs[0], mask)
        
        return outputs

    def mean(self, x):
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)
        
        return self.ssn_heads[0](outputs[0], mask)['logit_mean']

    def _samples_from_dist(self, dist, num_samples, rsample=True, symmetric=True):
        if symmetric:
            assert num_samples % 2 == 0
            num_samples = num_samples // 2

        samples = dist.rsample((num_samples,)) if rsample else dist.sample((num_samples,))
        
        if symmetric:
            mean = dist.mean
            samples = samples - mean
            return torch.cat([samples, -samples]) + mean
        else:
            return samples
    
    def mean_and_sample(self, x, num_samples, rsample=True, symmetric=True):
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)
        out = self.ssn_head(outputs[0], mask)
        mean = out['logit_mean']
        dist = out['distribution']
        samples = self._samples_from_dist(dist, num_samples, rsample, symmetric)
        return mean, samples


class SimpleRefinementHead(nn.Module):
    """
    A simple refinement head that takes concatenated features (final decoder + upsampled dist params)
    and outputs final distribution parameters through a few conv layers.
    """
    def __init__(self, 
                 in_channels, 
                 ssn_out_channels,
                 intermediate_channels,
                 out_channels, 
                 dims,
                 ):
        super().__init__()
        self.out_channels = out_channels
        
        Conv = nn.Conv2d if dims == 2 else nn.Conv3d
        Norm = nn.InstanceNorm2d if dims == 2 else nn.InstanceNorm3d

        # A small sequence of conv layers to refine features
        # Feel free to add more depth or complexity
        self.refine = nn.Sequential(
            Conv(in_channels + ssn_out_channels, intermediate_channels, kernel_size=3, padding=1),
            Norm(intermediate_channels),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            Norm(intermediate_channels),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            Norm(intermediate_channels),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(intermediate_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, output_head, spatial_sample):
        # print(spatial_sample.shape)
        # print(output_head.shape)
        spatial_sample = F.interpolate(spatial_sample, size=output_head.shape[2:], mode='nearest')
        x = torch.cat([output_head, spatial_sample], dim=1)
        
        return self.refine(x)


class SmallSpatialAttentionHead(nn.Module):
    """
    A simple refinement head that takes concatenated features (final decoder + upsampled dist params)
    and outputs final distribution parameters through a few conv layers.
    """
    def __init__(self, 
                 in_channels,
                 intermediate_channels,
                 out_channels, 
                 dims,
                 ):
        super().__init__()
        self.out_channels = out_channels
        
        Conv = nn.Conv2d if dims == 2 else nn.Conv3d
        Norm = nn.InstanceNorm2d if dims == 2 else nn.InstanceNorm3d

        self.spatial_attention_head = nn.Sequential(
            Conv(out_channels, 1, kernel_size=3, padding=1),
            Norm(1),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # A small sequence of conv layers to refine features
        # Feel free to add more depth or complexity
        self.refine = nn.Sequential(
            Conv(in_channels, intermediate_channels, kernel_size=3, padding=1),
            Norm(intermediate_channels),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            Norm(intermediate_channels),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(intermediate_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, output_head, spatial_sample):
        spatial_sample = F.interpolate(spatial_sample, size=output_head.shape[2:], mode='nearest')
        spatial_sample = self.spatial_attention_head(spatial_sample)
        spatial_sample = torch.sigmoid(spatial_sample)

        # print(spatial_sample.shape)
        # print(output_head.shape)
        
        return self.refine(output_head * spatial_sample)


class Hierarchical_SSN_with_ConvRefine(nn.Module):
    def __init__(self, 
                 base_unet, 
                 ssn_config,
                 refine_channels):
        """
        base_unet: model returning multiple decoder scales: [coarse_feats, ..., final_feats]
        coarse_level: index of the decoder output to use for the coarse distribution
        ssn_config_coarse: dict for coarse SSN head
        ssn_config_final: dict for final refinement head
        """
        super().__init__()
        self.base_unet = base_unet

        from twaibrain.braintorch.models.ssn import SSN_Head
        self.coarse_head = SSN_Head(
            intermediate_channels=ssn_config['intermediate_channels'],
            out_channels=ssn_config['out_channels'],
            dims=ssn_config['dims'],
            rank=ssn_config['rank'],
            diagonal=ssn_config.get('diagonal', False),
            epsilon=ssn_config.get('epsilon', 1e-5),
            return_cpu_dist=ssn_config.get('return_cpu_dist', False),
        )
        
        self.refine_head = SimpleRefinementHead(
            in_channels=ssn_config['intermediate_channels'], 
            ssn_out_channels=ssn_config['out_channels'],
            intermediate_channels=refine_channels,
            out_channels=ssn_config['out_channels'],
            dims=ssn_config['dims'],
        )

    def forward(self, x, symmetric=True, num_samples=10, rsample=True):
        # get main outputs from unet
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)

        # get the standard segmentation output for stage 2 of the decoder
        decoder_s2_out = outputs[2]
        mask_l = F.interpolate(mask, size=decoder_s2_out.shape[2:], mode='nearest')
        coarse_ssn = self.coarse_head(decoder_s2_out, mask_l)
        decoder_s2_mean = coarse_ssn['logit_mean']
        outputs[2] = decoder_s2_mean

        # get the distribution samples to propagate along
        dist = coarse_ssn['distribution']
        if symmetric:
            assert num_samples % 2 == 0
            half = num_samples // 2
            samples = dist.rsample((half,)) if rsample else dist.sample((half,))
            samples = torch.cat([samples - dist.mean, -(samples - dist.mean)]) + dist.mean
        else:
            samples = dist.rsample((num_samples,)) if rsample else dist.sample((num_samples,))

        ### pass the mean segmentation and the samples through the refinement head
        head_outputs = []
        mean_head_out = self.refine_head(outputs[0], decoder_s2_mean)
        sample_outputs = torch.stack([self.refine_head(outputs[0], s) for s in samples])

        outputs[0] = (mean_head_out, sample_outputs)

        return outputs


class Hierarchical_SSN_with_ConvSpatialAttention(nn.Module):
    def __init__(self, 
                 base_unet, 
                 ssn_config,
                 refine_channels):
        """
        base_unet: model returning multiple decoder scales: [coarse_feats, ..., final_feats]
        coarse_level: index of the decoder output to use for the coarse distribution
        ssn_config_coarse: dict for coarse SSN head
        ssn_config_final: dict for final refinement head
        """
        super().__init__()
        self.base_unet = base_unet

        from twaibrain.braintorch.models.ssn import SSN_Head
        self.coarse_head = SSN_Head(
            intermediate_channels=ssn_config['intermediate_channels'],
            out_channels=ssn_config['out_channels'],
            dims=ssn_config['dims'],
            rank=ssn_config['rank'],
            diagonal=ssn_config.get('diagonal', False),
            epsilon=ssn_config.get('epsilon', 1e-5),
            return_cpu_dist=ssn_config.get('return_cpu_dist', False),
        )
        
        self.refine_head = SmallSpatialAttentionHead(
            in_channels=ssn_config['intermediate_channels'], 
            intermediate_channels=refine_channels,
            out_channels=ssn_config['out_channels'],
            dims=ssn_config['dims'],
        )

    def forward(self, x, symmetric=True, num_samples=10, rsample=True):
        # get main outputs from unet
        outputs = self.base_unet(x)
        mask = x[:, -1].unsqueeze(1)

        # get the standard segmentation output for stage 2 of the decoder
        decoder_s2_out = outputs[2]
        mask_l = F.interpolate(mask, size=decoder_s2_out.shape[2:], mode='nearest')
        coarse_ssn = self.coarse_head(decoder_s2_out, mask_l)
        decoder_s2_mean = coarse_ssn['logit_mean']
        outputs[2] = decoder_s2_mean

        # get the distribution samples to propagate along
        dist = coarse_ssn['distribution']
        if symmetric:
            assert num_samples % 2 == 0
            half = num_samples // 2
            samples = dist.rsample((half,)) if rsample else dist.sample((half,))
            samples = torch.cat([samples - dist.mean, -(samples - dist.mean)]) + dist.mean
        else:
            samples = dist.rsample((num_samples,)) if rsample else dist.sample((num_samples,))

        ### pass the mean segmentation and the samples through the refinement head
        head_outputs = []
        mean_head_out = self.refine_head(outputs[0], decoder_s2_mean)
        sample_outputs = torch.stack([self.refine_head(outputs[0], s) for s in samples])

        outputs[0] = (mean_head_out, sample_outputs)

        return outputs

    def mean_and_sample(self, x, num_samples=10, rsample=True, symmetric=True):
        return self(x, symmetric=symmetric, num_samples=num_samples, rsample=rsample)[0]
