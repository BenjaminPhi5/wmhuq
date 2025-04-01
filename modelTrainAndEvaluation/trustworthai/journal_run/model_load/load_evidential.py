from trustworthai.models.core_models.Hypermapp3r import HyperMapp3r
from trustworthai.models.stochastic_wrappers.deterministic import Deterministic
from trustworthai.utils.losses_and_metrics.brier import get_brier_loss
from trustworthai.utils.losses_and_metrics.xent import xent_loss, mean_sum_xent_loss, dice_xent_loss
from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices
from trustworthai.utils.losses_and_metrics.evidential_bayes_risks import combined_evid_loss
import torch

def load_evidential(args):
    # for SSN and hypermapp3r, for now we use the number of first layer
    # channels as the number of out channels as well, prior to the ssn layer
    # which is of size 32 so maybe we should use 32?
    encoder_sizes=[16,32,64,128,256]
    base_model = HyperMapp3r(
        dropout_p=args.dropout_p,
        encoder_sizes=encoder_sizes,
        inchannels=3,
        outchannels=2
    )
    
    model_raw = Deterministic(base_model).cuda()
    
    loss = combined_evid_loss(args.dice_factor, args.xent_factor * args.xent_reweighting, args.kl_factor * args.xent_reweighting, anneal=True, anneal_count=args.kl_anneal_count, use_mle=args.use_mle)
    val_loss = combined_evid_loss(args.dice_factor, args.xent_factor * args.xent_reweighting, args.kl_factor * args.xent_reweighting, anneal=False, use_mle=args.use_mle)
    
    return model_raw, loss, val_loss