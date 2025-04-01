import torch
import torch.nn as nn
from trustworthai.models.stochastic_wrappers.ssn.LowRankMVCustom import LowRankMultivariateNormalCustom
from trustworthai.models.stochastic_wrappers.ssn.ReshapedDistribution import ReshapedDistribution
from trustworthai.models.uq_model import UncertaintyQuantificationModel
from tqdm import tqdm
import torch.distributions as td

class SSN(UncertaintyQuantificationModel):
    def __init__(self, base_model, rank, diagonal, epsilon, intermediate_channels, out_channels, dims, return_cpu_dist=False):
        super().__init__()
        self.base_model = base_model
        self.ssn_rank = rank
        self.ssn_diagonal = diagonal
        self.ssn_epsilon = epsilon
        self.ssn_num_classes = out_channels
        self.return_cpu_dist = return_cpu_dist
        
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.mean_l = nn.Conv2d(intermediate_channels, out_channels, kernel_size = (1,) *  dims, padding='same')
        self.log_cov_diag_l = nn.Conv2d(intermediate_channels, out_channels, kernel_size = (1,) * dims, padding='same')
        self.cov_factor_l = nn.Conv2d(intermediate_channels, out_channels * self.ssn_rank, kernel_size = (1,) * dims, padding='same')
        
    def forward(self, x):
        logits = self.lrelu(self.base_model(x))

        batch_size = logits.shape[0]
        event_shape = (self.ssn_num_classes,) + logits.shape[2:]
        
        mean = self.mean_l(logits)
        mean = mean.view((batch_size, -1))
        
        cov_diag = self.log_cov_diag_l(logits).exp() + self.ssn_epsilon
        cov_diag = cov_diag.view((batch_size, -1))
        
        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.ssn_rank, self.ssn_num_classes, -1))
        cov_factor = cov_factor.flatten(2,3)
        cov_factor = cov_factor.transpose(1,2)
        
        # covariance tends to blow up to infinity, hence set to 0 outside the ROI
        mask = x[:,-1]
        mask = mask.unsqueeze(1).expand((batch_size, self.ssn_num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_diag = cov_diag * mask + self.ssn_epsilon
    
        if self.return_cpu_dist:
            mean = mean.cpu()
            cov_diag = cov_diag.cpu()
            cov_factor = cov_factor.cpu()
            return mean, cov_diag, cov_factor, event_shape
        
        if self.ssn_diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = LowRankMultivariateNormalCustom(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except Exception as e:
                print("was thrown: ", e)
                print('hmm: Covariance became non invertible using independent normals for this batch!')
                print("cov diag okay: ", torch.sum(cov_diag <=0))
                print("sqrt cov diag okay: ", torch.sum(torch.sqrt(cov_diag) <=0))
                
                try:
                    base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)),1)
                except Exception as e:
                    print("second fail: ", e)
                    print(torch.min(torch.sqrt(cov_diag), torch.max(torch.sqrt(cov_diag))))
        
        distribution = ReshapedDistribution(base_distribution, event_shape)
        
        shape = (batch_size,) + event_shape
        logit_mean_view = mean.view(shape).detach()
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = cov_factor.transpose(2,1).view((batch_size, self.ssn_num_classes * self.ssn_rank) + event_shape[1:]).detach()
        
        output_dict = {
            'logit_mean':logit_mean_view,
            'cov_diag':cov_diag_view,
            'cov_factor':cov_factor_view,
            'distribution':distribution,
        }
        
        return output_dict
    
    def mean(self, x, temperature=1):
        return self(x)['logit_mean'] / temperature
    
    def _samples_from_dist(self, dist, num_samples, rsample=True, symmetric=True):
        if symmetric:
            assert num_samples % 2 == 0
            num_samples = num_samples // 2
            
        if rsample:
            samples = dist.rsample((num_samples,))
        else:
            samples = dist.sample((num_samples,))
        
        if symmetric:
            mean = dist.mean
            samples = samples - mean
            return torch.cat([samples, -samples]) + mean
        else:
            return samples
    
    def mean_and_sample(self, x, num_samples, rsample=True, temperature=1, symmetric=True):
        # NOTE: this does temperature scaling!!
        t = temperature
        out = self(x)
        mean = out['logit_mean']
        dist = out['distribution']
        samples = self._samples_from_dist(dist, num_samples, rsample, symmetric)
        return mean/t, samples/t
        
            