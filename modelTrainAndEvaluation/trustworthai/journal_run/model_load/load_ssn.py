from trustworthai.utils.losses_and_metrics.ssn_losses import SSNCombinedDiceXentLoss
from trustworthai.models.core_models.Hypermapp3r import HyperMapp3r
from trustworthai.models.stochastic_wrappers.ssn.ssn import SSN

def load_ssn(args):
    # for SSN and hypermapp3r, for now we use the number of first layer
    # channels as the number of out channels as well, prior to the ssn layer
    # which is of size 32 so maybe we should use 32?
    encoder_sizes=[16,32,64,128,256]
    base_model = HyperMapp3r(
        dropout_p=args.dropout_p,
        encoder_sizes=encoder_sizes,
        inchannels=3,
        outchannels=args.ssn_pre_head_layers
    )
    
    # wrap the base model in the SSN head
    diagonal = args.ssn_rank == 1
    model_raw = SSN(
        base_model=base_model,
        rank=args.ssn_rank,
        diagonal=diagonal,
        epsilon=args.ssn_epsilon,
        intermediate_channels=args.ssn_pre_head_layers,
        out_channels=2,
        dims=2, # 2D
    ).cuda()

    # define the loss function specific to SSN
    loss = SSNCombinedDiceXentLoss(
        empty_slice_weight=args.dice_empty_slice_weight,
        mc_samples=args.ssn_mc_samples,
        dice_factor=args.dice_factor,
        xent_factor=args.xent_factor * args.xent_reweighting,
        sample_dice_coeff=args.ssn_sample_dice_coeff,
    )
    
    return model_raw, loss, loss