### functions for getting code from the evidential distribution. Nice.
import torch
import torch.nn as nn
from torchmetrics import Metric


def relu_evidence(logits):
    return torch.nn.functional.relu(logits)

def exp_evidence(logits):
    return logits.clamp(-10, 10).exp()

def softplus_evidence(logits):
    return torch.nn.functional.softplus(logits)


def get_S(alpha):
    # evidence is shape [b, c, <dims>], we want an S per pixel, so reduce on dim 1
    S = alpha.sum(dim = 1).unsqueeze(1)
    return S

def get_bk(evidence, S):
    return evidence / S

def get_uncert(K, S):
    return K / S

def get_alpha(evidence):
    return (evidence + 1.)**2

def get_one_hot_target(K, target):
    one_hot = torch.zeros((target.shape[0], K, *target.shape[1:])).to(target.device)
    one_hot[:,0] = 1 - target
    one_hot[:,1] = target
    
    return one_hot

def get_mean_p_hat(alpha, S):
    return alpha / S

######
def digamma(values):
    return torch.digamma(values).clamp(-100,100)

def get_alpha_modified(alpha, one_hot_target):
    return one_hot_target + ((1 - one_hot_target) * alpha)

def MLE(alpha, S, one_hot_target):
    log_S = torch.log(S).expand(alpha.shape)
    log_alpha = torch.log(alpha)
    
    p_ij = one_hot_target * (log_S - log_alpha)
    per_pixel_loss =  torch.sum(p_ij, dim=1)
    
    # return torch.sum(per_pixel_loss, dim=(-2,-1)).mean() # reduction = batch mean
    return torch.mean(per_pixel_loss)  # reduction = pixel mean


def xent_bayes_risk(alpha, S, one_hot_target):
    digamma_S = torch.digamma(S).expand(alpha.shape)
    digamma_alpha = torch.digamma(alpha)
    
    p_ij = one_hot_target * (digamma_S - digamma_alpha)
    per_pixel_loss =  torch.sum(p_ij, dim=1)
    
    # return torch.sum(per_pixel_loss, dim=(-2,-1)).mean() # reduction = batch mean
    return torch.mean(per_pixel_loss)  # reduction = pixel mean


def mse_bayes_risk(mean_p_hat, S, one_hot_target):
    l_err = torch.nn.functional.mse_loss(mean_p_hat, one_hot_target, reduction='none')
    
    l_var = mean_p_hat * (1.- mean_p_hat) / (S + 1.)
    
    return (l_err + l_var).sum(dim=(-2,-1)).mean()

class KL_Loss():
    def __init__(self, anneal=True, anneal_count=452*4):
        self.counter = 0
        self.anneal = anneal
        self.anneal_count = anneal_count
        
    def __call__(self,alpha_modified):
        self.counter += 1
        #print(self.counter)
        K = alpha_modified.shape[1]
        beta = torch.ones((1, *alpha_modified.shape[1:])).to(alpha_modified.device)
        sum_alpha = alpha_modified.sum(dim=1)
        sum_beta = beta.sum(dim=1)

        lnB = torch.lgamma(sum_alpha) - torch.lgamma(alpha_modified).sum(dim=1)
        lnB_uni = torch.lgamma(beta).sum(dim=1) - torch.lgamma(sum_beta)

        dg0 = torch.digamma(sum_alpha).unsqueeze(1)
        dg1 = torch.digamma(alpha_modified)

        diff = (alpha_modified - beta)
        v = (dg1 - dg0)

        # print(sum_alpha.shape)
        # print(sum_beta.shape)
        # print(diff.shape)
        # print(v.shape)

        rhs = torch.sum(diff * v, dim=1)

        kl = lnB + lnB_uni + rhs

        # the early stopping checks val loss every three epochs. There are about 150
        # batches per train iteration currently, so 150*4, anneal in 4 epochs.
        
        loss = torch.mean(kl)
        # loss = torch.mean(kl, dim=(-2,-1)).mean()
        if self.anneal:
            # return torch.sum(kl, dim=(-2,-1)).mean() * (min(1, self.counter/(150*4))**2)
            anneal_factor = (min(1, self.counter/(150*4))**2)
            # print(f"kl loss annealed: {loss * anneal_factor:.3f}, factor: {anneal_factor:.3f} orig loss {loss:.3f}")
            return loss * anneal_factor
        else:
            # return torch.sum(kl, dim=(-2,-1)).mean()
            # print("kl not annealed")
            return loss
        

def dice_bayes_risk(K, alpha, one_hot_target, S):
    bs = alpha.shape[0]
    alpha = alpha.view(bs, K, -1)
    one_hot_target = one_hot_target.view(bs, K, -1)
    S = S.view(bs, 1, -1)
    #print(one_hot_target.shape, alpha.shape, S.shape)
    numerator = torch.sum(one_hot_target * alpha / S, dim=2)
    denominator = torch.sum(one_hot_target ** 2 + (alpha/S)**2 + (alpha*(S-alpha)/((S**2)*(S+1))), dim=2)
    
    dice = 1 - (2/K) * ((numerator/denominator).sum(dim=1))
    #print(dice.shape)
    return dice.mean()

def dice_bayes_risk_with_emtpty_slices(K, alpha, one_hot_target, S, empty_slice_weight):
    bs = alpha.shape[0]
    alpha = alpha.view(bs, K, -1)
    one_hot_target = one_hot_target.view(bs, K, -1)
    S = S.view(bs, 1, -1)
    #print(one_hot_target.shape, alpha.shape, S.shape)
    numerator = torch.sum(one_hot_target * alpha / S, dim=2)
    denominator = torch.sum(one_hot_target ** 2 + (alpha/S)**2 + (alpha*(S-alpha)/((S**2)*(S+1))), dim=2)
    
    if empty_slice_weight == 1:
        dice = 1 - (2/K) * ((numerator/denominator).sum(dim=1))
        #print(dice.shape)
        return dice.mean()
    
    else:
        # finding the empties
        locs = torch.sum(one_hot_target[:,1], dim=1) == 0
        #print(torch.sum(one_hot_target[:,1], dim=(-2, -1)), locs)
        wheres = torch.where(locs)[0]
        combined = (numerator/denominator)
        combined[wheres] *= empty_slice_weight
        #print(wheres)
        ratio = ((one_hot_target.shape[0] - wheres.shape[0]) + (wheres.shape[0] * empty_slice_weight))
        #print(ratio)
        
        dice_frac = (2/K) * combined.sum(dim=1)

        return  (1 - dice_frac.sum()/ratio)

#################### final combined loss functions and metrics
class combined_evid_loss(nn.Module):
    def __init__(self, dice_factor, xent_factor, kl_factor, anneal=True, anneal_count=452*4, use_mle=0):
        super().__init__()
        self.kl_obj = KL_Loss(anneal, anneal_count)
        self.dice_factor = dice_factor
        self.xent_factor = xent_factor
        self.kl_factor = kl_factor
        self.use_mle = use_mle
        #self.dice_empty_slice_weight = dice_empty_slice_weight
        
    def forward(self, logits, target):
        # get relevent terms required for loss func
        evidence = softplus_evidence(logits)
        alpha = get_alpha(evidence)
        S = get_S(alpha)
        K = alpha.shape[1]
        one_hot = get_one_hot_target(K, target)
        mean_p_hat = get_mean_p_hat(alpha, S)
        alpha_modified = get_alpha_modified(alpha, one_hot)


        #mse = mse_bayes_risk(mean_p_hat, S, one_hot)
        if self.use_mle == 1:
            xent = MLE(alpha, S, one_hot)
        else:
            xent = xent_bayes_risk(alpha, S, one_hot)
        # dice = dice_bayes_risk(K, alpha, one_hot, S, self.dice_empty_slice_weight)
        dice = dice_bayes_risk(K, alpha, one_hot, S)
        kl = self.kl_obj(alpha_modified)
        
        # print(f"dice loss {dice:.3f} xent_loss {xent:.3f}")

        total_loss = (dice * self.dice_factor) + (self.kl_factor * kl) + (xent * self.xent_factor)
        # print(f"dsc, kl, xent, {(dice * self.dice_factor):.3f} + {(self.kl_factor * kl):.3f} + {(xent * self.xent_factor):.3f}")
        return total_loss
    

# # metric that calcualtes the kl fully annealled, for early stopping
# class FullKLEvidMetric(Metric):
#     is_differentiable = False
#     higher_is_better = True
#     full_state_update = False
#     def __init__(self):
#         super().__init__()
#         self.add_state("value", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx=None)
#         self.kl_obj = KL_Loss(anneal=False)

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         self.value =  combined_loss(preds, target, self.kl_obj)

#     def compute(self):
#         return self.value