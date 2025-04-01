print("strawberry")

# loss function and metrics
from trustworthai.utils.losses_and_metrics.dice_loss import DiceLossWithWeightedEmptySlices
from trustworthai.utils.losses_and_metrics.dice_loss_metric import DiceLossMetric, SsnDiceMeanMetricWrapper

# predefined training dataset
from trustworthai.utils.data_preprep.dataset_pipelines import load_data

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.fitting_and_inference.fitters.p_unet_fitter import PUNetLitModelWrapper
from trustworthai.utils.fitting_and_inference.get_trainer import get_trainer

# model
from trustworthai.journal_run.model_load.load_deterministic import load_deterministic
from trustworthai.journal_run.model_load.load_evidential import load_evidential
from trustworthai.journal_run.model_load.load_punet import load_p_unet
from trustworthai.journal_run.model_load.load_ssn import load_ssn

# optimizer and lr scheduler
import torch

# misc
import argparse
import os
import shutil

print("banana")

MODEL_LOADERS = {
    "deterministic":load_deterministic,
    "mc_drop":load_deterministic,
    "evidential":load_evidential,
    "ssn":load_ssn,
    "punet":load_p_unet,
}

VOXELS_TO_WMH_RATIO = 382
VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES = 140

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--model_name', default=None, type=str)
    
    # data generation arguments
    parser.add_argument('--dataset', default='ed', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--test_split', default=0.15, type=float)
    parser.add_argument('--val_split', default=0.15, type=float)
    parser.add_argument('--empty_slice_retention', default=0.1, type=float)
    
    # select the model type to train
    parser.add_argument('--model_type', default="deterministic", type=str)
    
    # general arguments for the loss function
    parser.add_argument('--loss_name', default='dice+xent', type=str)
    parser.add_argument('--dice_factor', default=1, type=float) # 5
    parser.add_argument('--xent_factor', default=1, type=float) # 0.01
    parser.add_argument('--xent_reweighting', default=None, type=float)
    parser.add_argument('--xent_weight', default="none", type=str)
    parser.add_argument('--dice_empty_slice_weight', default=0.5, type=float)
    parser.add_argument('--tversky_beta', default=0.7, type=float)
    parser.add_argument('--reduction', default='mean_sum', type=str)
    
    # evidential arguments
    parser.add_argument('--kl_factor', default=0.1, type=float)
    parser.add_argument('--kl_anneal_count', default=452*4, type=int)
    parser.add_argument('--use_mle', default=0, type=int)
    parser.add_argument('--analytic_kl', default=0, type=int)
    
    # p-unet arguments
    parser.add_argument('--kl_beta', default=10.0, type=float)
    parser.add_argument('--use_prior_for_dice', default="false", type=str)
    parser.add_argument('--punet_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--latent_dim', default=12, type=int)
    
    # ssn arguments
    parser.add_argument('--ssn_rank', default=15, type=int)
    parser.add_argument('--ssn_epsilon', default=1e-5, type=float)
    parser.add_argument('--ssn_mc_samples', default=10, type=int)
    parser.add_argument('--ssn_sample_dice_coeff', default=0.05, type=float)
    parser.add_argument('--ssn_pre_head_layers', default=16, type=int)
    
    # training paradigm arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--dropout_p', default=0.0, type=float)
    parser.add_argument('--encoder_dropout1', default=0, type=int)
    parser.add_argument('--encoder_dropout2', default=0, type=int)
    parser.add_argument('--decoder_dropout1', default=0, type=int)
    parser.add_argument('--decoder_dropout2', default=0, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=15, type=int)
    parser.add_argument('--scheduler_type', default='step', type=str)
    parser.add_argument('--scheduler_step_size', default=1000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.5, type=float)
    parser.add_argument('--scheduler_power', default=0.9, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cross_validate', default="true", type=str)
    parser.add_argument('--cv_split', default=0, type=int)
    parser.add_argument('--cv_test_fold_smooth', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--overwrite', default="false", type=str)
    parser.add_argument('--no_test_fold', default='false', type=str)
    
    return parser


def main(args):
    
    # sanitise arguments
    args.overwrite = True if args.overwrite.lower() == "true" else False
    args.cross_validate = True if args.cross_validate.lower() == "true" else False
    args.use_prior_for_dice = True if args.use_prior_for_dice.lower() == "true" else False
    print(args)
    try:
        model_loader = MODEL_LOADERS[args.model_type]
    except:
        raise ValueError(f"argument model_type should be one of {MODEL_LOADERS.keys()} not {args.model_type}")
        
    # deal with scratch / scratch big issue
    if "scratch/" in args.ckpt_dir:
        if not os.path.exists("/disk/scratch/"):
            args.ckpt_dir = args.ckpt_dir.replace("scratch", "scratch_big")
            
    
    # setup xent reweighting factor
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    args.xent_reweighting = XENT_WEIGHTING
    
    # for now the model name is passed directly from the expierment script
    # which means I need to manually configure the model names but its not the end of the world
    # a better system would be nice though
    model_dir = os.path.join(args.ckpt_dir, args.model_name) 
    print("model dir: ", model_dir)
    
    # check that the requested model doesn't exist anywhere.
    if os.path.exists(model_dir):
        if not args.overwrite:
            raise ValueError(f"model directly ALREADY EXISTS: do not wish to overwrite!!: {model_dir}")
        else:
            print("warning, folder being overwritten")
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
    
    # get the 2d axial slice dataloaders
    train_dl, val_dl, test_dl = load_data(
        dataset=args.dataset, 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataloader2d_only=True,
        cross_validate=args.cross_validate,
        cv_split=args.cv_split,
        cv_test_fold_smooth=args.cv_test_fold_smooth,
        merge_val_test=args.no_test_fold
    )
    
    # load the model and loss function
    model_raw, loss, val_loss = model_loader(args)

    # setup optimizer and model wrapper
    optimizer_params={"lr":args.lr, "weight_decay":args.weight_decay}
    optimizer = torch.optim.Adam
    if args.scheduler_type == "step":
        lr_scheduler_constructor = torch.optim.lr_scheduler.StepLR
        lr_scheduler_params = {"step_size":args.scheduler_step_size, "gamma":args.scheduler_gamma}
    elif args.scheduler_type == "poly":
        lr_scheduler_constructor = torch.optim.lr_scheduler.PolynomialLR
        lr_scheduler_params = {"power":args.scheduler_power, "total_iters":args.max_epochs}
    elif args.scheduler_type == "multistep":
        lr_scheduler_constructor = torch.optim.lr_scheduler.MultiStepLR
        milestones = [i * args.scheduler_step_size for i in range(1, 5)]
        print(f"milestones: {milestones}")
        lr_scheduler_params = {"milestones":milestones, "gamma":args.scheduler_gamma}
    else:
        raise ValueError

    # wrap the model in the pytorch_lightning module that automates training
    if args.model_type == "punet":
        extra_args = {"use_prior_for_dice":args.use_prior_for_dice}
        model_wrapper = PUNetLitModelWrapper
    else:
        extra_args = {}
        model_wrapper = StandardLitModelWrapper
        
    model = model_wrapper(model_raw, loss,
                                logging_metric=lambda : None,
                                val_loss=val_loss,
                                optimizer_params=optimizer_params,
                                lr_scheduler_params=lr_scheduler_params,
                                optimizer_constructor=optimizer,
                                lr_scheduler_constructor=lr_scheduler_constructor,
                               **extra_args)
   
    # train the model
    trainer = get_trainer(args.max_epochs, model_dir, early_stop_patience=args.early_stop_patience)
    trainer.fit(model, train_dl, val_dl)
    
    # get best checkpoint based on loss on validation data
    try:
        #"save best model checkpoint name"
        with open(os.path.join(model_dir, "best_ckpt.txt"), "w") as f:
            f.write(trainer.checkpoint_callback.best_model_path)
            f.write("\n")
            for key , value in vars(args).items():
                f.write(f"{key}: {value}\n")
            
        trainer.validate(model, val_dl, ckpt_path='best')
    except:
        print("failed to run validate to print best checkpoint path oh well")


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
