import torch
import torch.nn.functional as F
import torch.nn as nn
from twaibrain.braintorch.losses.dice_loss import SoftDiceV2
from twaibrain.braintorch.losses.xent import xent_loss
import math
    
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

class SSN_SoftDice_Loss_Mean_and_Samples(nn.Module):
    def __init__(self, mean_weight, samples_weight):
        super().__init__()
        self.dice_loss = SoftDiceV2()
        self.mean_weight = mean_weight
        self.samples_weight = samples_weight

    def forward(self, mean, samples, target):
        loss = self.dice_loss(mean, target) * self.mean_weight
        Ns = samples.shape[0]
        for s in samples:
            loss += self.dice_loss(s, target) * self.samples_weight / Ns

        return loss

class SSN_SoftDice_Loss_Mean_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = SoftDiceV2()

    def forward(self, mean, samples, target):
        loss = self.dice_loss(mean, target)
        return loss

class SSN_SoftDice_Loss_Samples_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = SoftDiceV2()

    def forward(self, mean, samples, target):
        s_loss = self.dice_loss(samples[0], target)
        for s in samples[1:]:
            s_loss += self.dice_loss(s, target)

        s_loss /= samples.shape[0]

        return s_loss

class SSN_Best_SoftDice_Loss_Samples(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = SoftDiceV2()

    def forward(self, mean, samples, target):
        s_losses = [self.dice_loss(s, target) for s in samples]
        s_losses = torch.Tensor(s_losses)
        return torch.min(s_losses)

class SSN_Xent_Loss_Mean_and_Samples(nn.Module):
    def __init__(self, mean_weight, samples_weight):
        super().__init__()
        self.xent_loss = xent_loss(weight=None, reduction='mean')
        self.mc_loss = StochasticSegmentationNetworkLossMCIntegral_FromSamples()
        self.mean_weight = mean_weight
        self.samples_weight = samples_weight

    def forward(self, mean, samples, target):
        loss = self.xent_loss(mean, target) * self.mean_weight
        loss += self.mc_loss(samples, target) * self.samples_weight

        return loss

class SSN_Xent_loss_Mean_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = xent_loss(weight=None, reduction='mean')

    def forward(self, mean, samples, target):
        loss = self.xent_loss(mean, target)
        return loss

class SSN_Xent_Loss_Samples_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.mc_loss = StochasticSegmentationNetworkLossMCIntegral_FromSamples()

    def forward(self, mean, samples, target):
        return self.mc_loss(samples, target) 

class SSN_Best_Xent_Loss_Samples(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = xent_loss(weight=None, reduction='mean')

    def forward(self, mean, samples, target):
        s_losses = [self.xent_loss(s, target) for s in samples]
        s_losses = torch.Tensor(s_losses)
        return torch.min(s_losses)

def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

class SSN_ComboLoss(nn.Module):
    def __init__(self, dice_weight=1, xent_weight=1, mean_weight=0, sample_weight=1, dice_added_sample_weight=1, best_sample=False, mc_samples=10):
        super().__init__()
        self.dice_mean_loss_func = SSN_SoftDice_Loss_Mean_Only()
        self.dice_sample_loss_func = SSN_SoftDice_Loss_Samples_Only() if not best_sample else SSN_Best_SoftDice_Loss_Samples()

        self.xent_mean_loss_func = SSN_Xent_loss_Mean_Only()
        self.xent_sample_loss_func = SSN_Xent_Loss_Samples_Only() if not best_sample else SSN_Best_Xent_Loss_Samples()

        self.dice_weight = dice_weight
        self.xent_weight = xent_weight

        self.mean_weight = mean_weight
        self.sample_weight = sample_weight
        self.dice_added_sample_weight = dice_added_sample_weight

        self.num_mc_samples = mc_samples
    
    def forward(self, results, target):
        mean = results['logit_mean']
        samples = fixed_re_parametrization_trick(results['distribution'], self.num_mc_samples)
        
        mean_dice_loss = self.dice_mean_loss_func(mean, samples, target)
        samples_dice_loss = self.dice_sample_loss_func(mean, samples, target) * self.dice_added_sample_weight
        dice_loss = mean_dice_loss * self.mean_weight + samples_dice_loss * self.sample_weight

        mean_xent_loss = self.xent_mean_loss_func(mean, samples, target)
        samples_xent_loss = self.xent_sample_loss_func(mean, samples, target)
        xent_loss = mean_xent_loss * self.mean_weight + samples_xent_loss * self.sample_weight

        return dice_loss * self.dice_weight + xent_loss * self.xent_weight
        
class SSNDeepSupervisionLoss():
    def __init__(self, criterion, print_components=False):
        self.criterion = criterion
        self.print_components = print_components

    def __call__(self, outputs, target):
        main_output = outputs[0]
        deep_outputs = outputs[1:]
        loss = self.criterion(main_output, target.squeeze(dim=1).type(torch.long))
        if self.print_components:
            print(loss.item())
        
        # Apply deep supervision
        deep_supervision_weight = 0.5
        for deep_output in deep_outputs:
            # Resize target to match deep output size
            resized_target = F.interpolate(target.float(), size=deep_output['logit_mean'].shape[2:], mode='nearest')
            loss_component = self.criterion(deep_output, resized_target.squeeze(dim=1).type(torch.long))
            if self.print_components:
                print(loss_component.item())
            loss += deep_supervision_weight * loss_component
            deep_supervision_weight *= deep_supervision_weight
    
        return loss

class LLOMultiDeepSupervisionLoss():
    def __init__(self, criterion_ssn, criterion_other_layers, print_components=False):
        self.criterion_0 = criterion_ssn
        self.criterion_1 = criterion_other_layers
        self.print_components = print_components

    def __call__(self, outputs, target):
        main_output = outputs[0]
        deep_outputs = outputs[1:]
        loss = self.criterion_0(main_output, target.squeeze(dim=1).type(torch.long))
        if self.print_components:
            print(loss.item())
            
        # Apply deep supervision
        deep_supervision_weight = 0.5
        for deep_output in deep_outputs:
            # Resize target to match deep output size
            resized_target = F.interpolate(target.float(), size=deep_output.shape[2:], mode='nearest')
            loss_component = self.criterion_1(deep_output, resized_target.squeeze(dim=1).type(torch.long))
            if self.print_components:
                print(loss_component.item())
            loss += deep_supervision_weight * loss_component
            deep_supervision_weight *= deep_supervision_weight

        return loss
    