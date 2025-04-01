from trustworthai.models.core_models.Hypermapp3r import HyperMapp3r
from trustworthai.models.stochastic_wrappers.deterministic import Deterministic
from trustworthai.utils.losses_and_metrics.brier import get_brier_loss
from trustworthai.utils.losses_and_metrics.xent import xent_loss, mean_sum_xent_loss, dice_xent_loss
from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices, SoftDiceV2, SoftDiceV1
from trustworthai.utils.losses_and_metrics.tversky_loss import TverskyLoss
import torch

def load_deterministic(args):
    # for SSN and hypermapp3r, for now we use the number of first layer
    # channels as the number of out channels as well, prior to the ssn layer
    # which is of size 32 so maybe we should use 32?
    encoder_sizes=[16,32,64,128,256]
    base_model = HyperMapp3r(
        dropout_p=args.dropout_p,
        encoder_sizes=encoder_sizes,
        inchannels=3,
        outchannels=2,
        encoder_dropout1=args.encoder_dropout1,
        encoder_dropout2=args.encoder_dropout2,
        decoder_dropout1=args.decoder_dropout1,
        decoder_dropout2=args.decoder_dropout2,
    )
    
    # this line is redundant but removing it will require retraining
    # as the models won't load correctly anymore...
    model_raw = Deterministic(base_model).cuda()

    loss_name = args.loss_name
    weight_name = args.xent_weight
    weights = {
        "standard":[0.05, 1],
        "big":[0.03, 33],
        "none":[1,1],
        "inverse":[1,0.05],
    }
    weight = torch.Tensor(weights[weight_name])
    
    losses = {
        "dice":DiceLossWithWeightedEmptySlices(args.dice_empty_slice_weight),
        "dicev1":SoftDiceV1(),
        "dicev2":SoftDiceV2(),
        "brier":get_brier_loss(reduction=args.reduction),
        "xent":xent_loss(weight, args.reduction),
        "tversky":TverskyLoss(args.tversky_beta),
        "old_dice+xent":dice_xent_loss(
            DiceLossWithWeightedEmptySlices(args.dice_empty_slice_weight),
            xent_loss(weight, args.reduction),
            args.dice_factor,
            args.xent_factor
        ),
        "dice+xent":dice_xent_loss(
            SoftDiceV2(), xent_loss(weight=None, reduction="mean"),
            args.dice_factor,
            args.xent_factor * args.xent_reweighting,
        )
    }
    loss = losses[loss_name]
    
    return model_raw, loss, loss