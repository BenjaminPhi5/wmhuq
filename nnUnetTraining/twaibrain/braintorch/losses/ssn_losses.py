import torch
import torch.nn.functional as F
import torch.nn as nn
from twaibrain.braintorch.losses.dice_loss import DiceLossWithWeightedEmptySlices, SoftDiceV2
from twaibrain.braintorch.losses.xent import dice_xent_loss
import math
        
class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    def forward(self, result_dict, target, **kwargs):
        logits = result_dict['logit_mean']
        distribution = result_dict['distribution']
        
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        assert num_classes >= 2  # not implemented for binary case with implied background
        # logit_sample = distribution.rsample((self.num_mc_samples,))
        logit_sample = fixed_re_parametrization_trick(distribution, self.num_mc_samples)
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)

        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        n_voxels = logit_sample.shape[-1]
        target = target.reshape((flat_size, -1))

        log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view((self.num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples))
        loss = -loglikelihood
        return loss / n_voxels # I added / n_voxels to get the loss on the same scale as cross entropy is for the other methods.
    
class StochasticSegmentationNetworkLossMCIntegral_FromSamples(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, samples, target, **kwargs):
        num_mc_samples = samples.shape[0]
        batch_size = samples.shape[1]
        num_classes = samples.shape[2]
        assert num_classes >= 2  # not implemented for binary case with implied background
        # logit_sample = distribution.rsample((self.num_mc_samples,))
        logit_sample = samples
        target = target.unsqueeze(1)
        target = target.expand((num_mc_samples,) + target.shape)

        flat_size = num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        n_voxels = logit_sample.shape[-1]
        target = target.reshape((flat_size, -1))

        log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view((num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(num_mc_samples))
        loss = -loglikelihood
        return loss / n_voxels # I added / n_voxels to get the loss on the same scale as cross entropy is for the other methods.

    
def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean
    
    
class SsnNetworkMeanLossWrapper(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss = loss_func
    def forward(self, result_dict, target):
        mean = result_dict['logit_mean']
        return self.loss(mean, target)
    
class SsnNetworkSampleLossWrapper(nn.Module):
    def __init__(self, loss_func, samples=10):
        super().__init__()
        self.loss = loss_func
        self.samples = samples
    def forward(self, result_dict, target):
        samples = fixed_re_parametrization_trick(result_dict['distribution'], self.samples).to(target.device)
        loss = 0
        for s in samples:
            loss += self.loss(s, target)
        return loss / self.samples
    
class SsnNetworkMuAndSamplesLossWrapper(nn.Module):
    def __init__(self, loss_func, samples=10, sample_loss_coeff=0.05):
        super().__init__()
        self.loss = loss_func
        self.samples = samples
        self.sample_loss_coeff = sample_loss_coeff
        
    def forward(self, result_dict, target):
        s = result_dict['distribution'].mean
        
        dice = self.loss(s, target)
        samples = fixed_re_parametrization_trick(result_dict['distribution'], self.samples).to(target.device)
        loss = 0
        for s in samples:
            loss += self.loss(s, target)
        
        return (dice + ((self.sample_loss_coeff*loss) / self.samples)) / (1 + self.sample_loss_coeff)


class SSNCombinedDiceXentLoss(nn.Module):
    def __init__(self, empty_slice_weight=0.5, mc_samples=10, dice_factor=5, xent_factor=0.01, sample_dice_coeff=0.05):
        super().__init__()
        
        dice_loss = SoftDiceV2() # DiceLossWithWeightedEmptySlices(r=0.5)
        self.samples_dice_loss = SsnNetworkMuAndSamplesLossWrapper(dice_loss, samples=mc_samples, sample_loss_coeff=sample_dice_coeff)
        self.mc_loss = StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=mc_samples)
        self.dice_factor = dice_factor
        self.xent_factor = xent_factor
    
    def forward(self, result_dict, target):
        return self.samples_dice_loss(result_dict, target) * self.dice_factor + self.mc_loss(result_dict, target) * self.xent_factor


class SSNCombinedDiceXent_and_MC_loss(nn.Module):
    def __init__(self, mc_samples=10, dice_factor=5, xent_factor=0.01):
        super().__init__()
        
        self.combo_loss = dice_xent_loss(dice_weight=dice_factor, xent_weight=xent_factor)
        self.mc_loss = StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=mc_samples)
        self.dice_factor = dice_factor
        self.xent_factor = xent_factor
        self.mc_sample_ratio = (mc_samples - 1) / mc_samples
    
    def forward(self, result_dict, target):
        return self.combo_loss(result_dict['logit_mean'], target) + self.mc_loss(result_dict, target) * self.xent_factor * mc_sample_ratio


class SSNCombinedDiceXent_and_MC_loss_FromSamples(nn.Module):
    def __init__(self, dice_factor=5, xent_factor=0.01):
        super().__init__()
        
        self.combo_loss = dice_xent_loss(dice_weight=dice_factor, xent_weight=xent_factor)
        self.mc_loss = StochasticSegmentationNetworkLossMCIntegral_FromSamples()
        self.dice_factor = dice_factor
        self.xent_factor = xent_factor
    
    def forward(self, results, target):
        num_samples = results[1].shape[0]
        mc_sample_ratio = (num_samples - 1) / num_samples
        return self.combo_loss(results[0], target) + self.mc_loss(results[1], target) * self.xent_factor * mc_sample_ratio



class DeepSupervisionSSN(SSNCombinedDiceXentLoss):
    def forward(self, outputs, target, **kwargs):
        main_output = outputs[0]
        deep_outputs = outputs[1:]
        loss = super().forward(main_output, target)
        # print(loss)
        deep_supervision_weight = 0.5
        for o in deep_outputs:
            resized_target = F.interpolate(target.float(), size=o['logit_mean'].shape[2:], mode='nearest')
            loss += super().forward(o, resized_target.long()) * deep_supervision_weight
            deep_supervision_weight *= deep_supervision_weight
            # print(loss)
        return loss