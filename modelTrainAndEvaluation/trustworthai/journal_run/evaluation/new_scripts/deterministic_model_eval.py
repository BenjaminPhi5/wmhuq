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

# optimizer and lr scheduler
import torch

# misc
import argparse
import os
import pandas

# evaluation code
from trustworthai.journal_run.evaluation.new_scripts.eval_helper_functions import *

print("banana")

VOXELS_TO_WMH_RATIO = 382
VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES = 140

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    # folder arguments
    parser.add_argument('--ckpt_dir', default='s2208943/results/revamped_models/', type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--repo_dir', default=None, type=str)
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--eval_split', default='val', type=str)
    
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
    parser.add_argument('--cross_validate', default=False, type=bool)
    parser.add_argument('--cv_split', default=0, type=int)
    parser.add_argument('--cv_test_fold_smooth', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--overwrite', default="false", type=str)
    parser.add_argument('--no_test_fold', default='false', type=str)
    
    return parser


def main(args):
    
    # sanitise arguments
    args.overwrite = True if args.overwrite.lower() == "true" else False
    #print(args)
    
    if not args.overwrite:
        model_result_folder = os.path.join(args.repo_dir, args.result_dir)
        existing_files = os.listdir(model_result_folder)
        for f in existing_files:
            if args.model_name + "_" in f:
                raise ValueError(f"ovewrite = false and model results exist! folder={model_result_folder}, model_name={args.model_name}")
            else:
                with open(os.path.join(model_result_folder, f"{args.model_name}_init.txt"), "w") as f:
                          f.write("generating results\n")
        
    # setup xent reweighting factor
    XENT_VOXEL_RESCALE = VOXELS_TO_WMH_RATIO - (1-args.empty_slice_retention) * (VOXELS_TO_WMH_RATIO - VOXELS_TO_WMH_RATIO_EXCLUDING_EMPTY_SLICES)

    XENT_WEIGHTING = XENT_VOXEL_RESCALE/2
    args.xent_reweighting = XENT_WEIGHTING
    
    # for now the model name is passed directly from the expierment script
    # which means I need to manually configure the model names but its not the end of the world
    # a better system would be nice though
    model_dir = os.path.join(args.ckpt_dir, args.model_name) 
    print("model dir: ", model_dir)
    
    # check that the requested model exists
    if not os.path.exists(model_dir):
        raise ValueError(f"{model_dir} does not exist.")
    
    # get the 3d axial slice dataloaders
    data_dict = load_data(
        dataset=args.dataset, 
        test_proportion=args.test_split, 
        validation_proportion=args.val_split,
        seed=args.seed,
        empty_proportion_retained=args.empty_slice_retention,
        batch_size=args.batch_size,
        dataloader2d_only=False,
        cross_validate=args.cross_validate,
        cv_split=args.cv_split,
        cv_test_fold_smooth=args.cv_test_fold_smooth,
        merge_val_test=args.no_test_fold
    )
    
    eval_ds = data_dict[f'{args.eval_split}_dataset3d']
    
    # load the model and loss function
    model_loader = load_deterministic
    model_raw, loss, val_loss = model_loader(args)
    
    # load the checkpoint under consideration
    model_dir = os.path.join(args.ckpt_dir, args.model_name) 
    print("model dir: ", model_dir)
    try:
        model = load_best_checkpoint(model_raw, loss, model_dir)
    except:
        raise ValueError(f"dir {model_dir} doesn't contain checkpoints")
    
    # get the xs and ys
    xs3d_test = []
    ys3d_test = []

    for i, data in enumerate(eval_ds):
        ys3d_test.append(data[1].squeeze())
        xs3d_test.append(data[0])
    
    # load the mean predictions
    means = []
    temp = 1
    batch_image = False

    model.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True)):
        x = data[0]
        with torch.no_grad():
            if batch_image:
                mean = model(x.swapaxes(0,1).cuda()).cpu()
            else:
                mean = []
                for islice in range(x.shape[1]):
                    mean.append(model(x[:,islice].unsqueeze(0).cuda()).cpu())
                mean = torch.cat(mean)
            means.append(mean / temp)

    preds = [torch.softmax(m.cuda(), dim=1).cpu() for m in means]
    
    # run the evaluation
    write_per_model_channel_stats(preds, ys3d_test, args)

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
