from trustworthai.utils.losses_and_metrics.p_unet_loss import punet_loss
from trustworthai.models.core_models.Hypermapp3r import HyperMapp3r
from trustworthai.models.stochastic_wrappers.p_unet.p_unet import ProbabilisticUnet

def load_p_unet(args):
    encoder_sizes=[16,32,64,128,256]
    base_model = HyperMapp3r(
        dropout_p=args.dropout_p,
        encoder_sizes=encoder_sizes,
        inchannels=3,
        outchannels=2,
        p_unet_hook=True,
    )
    
    model = ProbabilisticUnet(base_model, input_channels=3, num_classes=2, num_filters=[16, 32, 64, 128, 256], latent_dim=args.latent_dim, no_convs_fcomb=4, beta=args.kl_beta)

    # kl_factor = 1/args.batch_size
    kl_factor = 1
    loss = punet_loss(args.xent_factor * args.xent_reweighting, args.dice_factor, args.punet_sample_dice_coeff, kl_factor, analytic_kl=args.analytic_kl)
    
    return model, loss, loss