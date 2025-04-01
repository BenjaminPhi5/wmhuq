import torch

def two_class_prob(p_hat):
    p_hat = torch.nn.functional.softmax(p_hat, dim=1)
    p_hat = p_hat[:,1] # select class 0
    return p_hat

def implicit_prob(p_hat):
    p_hat = torch.sigmoid(p_hat)
    return p_hat

def get_normalized_probs(p_hat):
    if p_hat.shape[1] == 1:
        return implicit_prob(p_hat)
    else:
        return two_class_prob(p_hat)

def individual_dice_v1(p_hat, y_true):
    p_hat = get_normalized_probs(p_hat)
    s0 = p_hat.shape[0]
    p_hat = p_hat.view(s0,-1)
    y_true = y_true.view(s0,-1)
    numerator = torch.sum(2. * p_hat * y_true, dim=1) + 1.
    denominator = torch.sum(y_true + p_hat, dim=1) + 1.
    combined = 1. - (numerator/denominator)
    return combined

def individual_dice_v2(p_hat, y_true):
    p_hat = get_normalized_probs(p_hat)
    s0 = p_hat.shape[0]
    p_hat = p_hat.view(s0,-1)
    y_true = y_true.view(s0,-1)
    numerator = torch.sum(2. * p_hat * y_true, dim=1) + 1.
    denominator = torch.sum(y_true.square() + p_hat.square(), dim=1) + 1.
    combined = 1. - (numerator/denominator)
    return combined
    
def dice_loss(p_hat, y_true, func=individual_dice_v1):
    combined = func(p_hat, y_true)
    return torch.mean(combined)

class SoftDiceV1:
    def __call__(self, p_hat, y_true):
        return dice_loss(p_hat, y_true, individual_dice_v1)
    
class SoftDiceV2:
    def __call__(self, p_hat, y_true):
        return dice_loss(p_hat, y_true, individual_dice_v2)

class DiceLossWithWeightedEmptySlices:
    def __init__(self, r=0.5):
        """
        r is the weight applied to empty slices
        implicit_class: true if model only outputs one final channel
        """
        self.r = r
    
    def __call__(self, p_hat, y_true):
        combined = individual_dice(p_hat, y_true)
    
        # is empties
        locs = torch.sum(y_true, dim=(-2, -1)) == 0
        wheres = torch.where(locs)[0]
        combined[wheres] *= self.r

        return (
            torch.sum(combined) 
            / ((y_true.shape[0] - wheres.shape[0]) + (wheres.shape[0] * self.r))
        )

    