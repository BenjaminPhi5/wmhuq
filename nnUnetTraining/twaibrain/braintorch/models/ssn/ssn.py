import torch
import torch.nn as nn
import torch.distributions as td
from twaibrain.braintorch.models.ssn import LowRankMultivariateNormalCustom, ReshapedDistribution

class SSN_Head(nn.Module):
    def __init__(
        self,
        intermediate_channels,
        out_channels,
        dims=2,
        rank=10,
        diagonal=False,
        epsilon=1e-5,
        return_cpu_dist=False
    ):
        super().__init__()
        self.ssn_rank = rank
        self.ssn_diagonal = diagonal
        self.ssn_epsilon = epsilon
        self.ssn_num_classes = out_channels
        self.return_cpu_dist = return_cpu_dist
        self.lrelu = nn.LeakyReLU(0.01)

        # Use appropriate conv type depending on dims
        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.mean_l = Conv(intermediate_channels, out_channels, kernel_size = (1,) *  dims, padding='same')
        self.log_cov_diag_l = Conv(intermediate_channels, out_channels, kernel_size = (1,) * dims, padding='same')
        self.cov_factor_l = Conv(intermediate_channels, out_channels * self.ssn_rank, kernel_size = (1,) * dims, padding='same')
    
    def forward(self, x, mask):
        # x is assumed to be the raw features from the decoder layer
        logits = self.lrelu(x)

        batch_size = logits.shape[0]
        spatial_shape = logits.shape[2:]  # everything except batch & channels
        event_shape = (self.ssn_num_classes,) + spatial_shape

        mean = self.mean_l(logits)  # shape: B, out_channels, ...
        mean = mean.view((batch_size, -1))
        

        cov_diag = (self.log_cov_diag_l(logits).exp() + self.ssn_epsilon)
        cov_diag = cov_diag.view((batch_size, -1))

        
        cov_factor = self.cov_factor_l(logits)  # B, out_channels*rank, ...
        cov_factor = cov_factor.view((batch_size, self.ssn_rank,  self.ssn_num_classes, -1))
        cov_factor = cov_factor.flatten(2,3)
        cov_factor = cov_factor.transpose(1, 2)
        
        mask = mask.expand((batch_size, self.ssn_num_classes) + mask.shape[2:])
        mask = mask.reshape(batch_size, -1)

        # print(mean.shape, mask.shape, cov_factor.shape, cov_diag.shape)

        cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_diag = cov_diag * mask + self.ssn_epsilon

        if self.return_cpu_dist:
            mean = mean.cpu()
            cov_diag = cov_diag.cpu()
            cov_factor = cov_factor.cpu()
            return mean, cov_diag, cov_factor, event_shape

        # Construct the distribution
        if self.ssn_diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = LowRankMultivariateNormalCustom(
                    loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                )
            except Exception as e:
                print("Covariance issue, fallback to Independent Normal. Error:", e)
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)
        shape = (batch_size,) + event_shape
        
        output_dict = {
            'logit_mean': mean.view(shape).detach(),
            'cov_diag': cov_diag.view(shape).detach(),
            'cov_factor': cov_factor.transpose(2,1).view(
                (batch_size, self.ssn_num_classes * self.ssn_rank) + event_shape[1:]
            ).detach(),
            'distribution': distribution,
        }
        return output_dict
    
    def mean(self, x):
        return self(x)['logit_mean']
    
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
        out = self(x)
        mean = out['logit_mean']
        dist = out['distribution']
        samples = self._samples_from_dist(dist, num_samples, rsample, symmetric)
        return mean, samples
