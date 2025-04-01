print("strawberry")

import torch
import numpy as np
import torch.nn.functional as F

# dataset
from twaidata.torchdatasets.in_ram_ds import MRISegmentation2DDataset, MRISegmentation3DDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

# model
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_deterministic import HyperMapp3r
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_DDU import HyperMapp3rDDU
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_SSN import HyperMapp3rSSN


# augmentation and pretrain processing
from trustworthai.utils.augmentation.standard_transforms import RandomFlip, GaussianBlur, GaussianNoise, \
                                                            RandomResizeCrop, RandomAffine, \
                                                            NormalizeImg, PairedCompose, LabelSelect, \
                                                            PairedCentreCrop, CropZDim
# loss function
from trustworthai.utils.losses_and_metrics.per_individual_losses import (
    log_cosh_dice_loss,
    TverskyLoss,
    FocalTverskyLoss,
    DiceLossMetric
)
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

# misc
import os
import torch
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import argparse

import torch.nn as nn
import torch
from torchmetrics import Metric
import math

def two_class_prob(p_hat):
    p_hat = torch.nn.functional.softmax(p_hat, dim=1)
    p_hat = p_hat[:,1,:] # select class 0
    return p_hat

def individual_dice(p_hat, y_true):
    p_hat = two_class_prob(p_hat)
    s0 = p_hat.shape[0]
    p_hat = p_hat.view(s0,-1)
    y_true = y_true.view(s0,-1)
    numerator = torch.sum(2. * p_hat * y_true, dim=1) + 1.
    denominator = torch.sum(y_true + p_hat, dim=1) + 1.
    combined = 1. - (numerator/denominator)
    return combined
    
def dice_loss(p_hat, y_true):
    combined = individual_dice(p_hat, y_true)
    
    # is empties
    locs = torch.sum(y_true, dim=(-2, -1)) == 0
    wheres = torch.where(locs)[0]
    #print(wheres.shape)
    # print(wheres)
    #print(combined)
    r = 0.5
    combined[wheres] *= r
    #print(combined)
    
    return torch.sum(combined) / ((y_true.shape[0] - wheres.shape[0]) + (wheres.shape[0] * r))

def dice_loss_old(p_hat, y_true):
    combined = individual_dice(p_hat, y_true)
    return torch.mean(combined)

def construct_parser():
    parser = argparse.ArgumentParser(description = "train models")
    
    parser.add_argument('--ckpt_dir', default=None, type=str)
    parser.add_argument('--dice_factor', default=None, type=int)
    return parser

def get_transforms(is_3D):
        transforms = [
            LabelSelect(label_id=1),
            RandomFlip(p=0.5, orientation="horizontal"),
            # GaussianBlur(p=0.5, kernel_size=7, sigma=(.1, 1.5)),
            # GaussianNoise(p=0.2, mean=0, sigma=0.2),
            # RandomAffine(p=0.2, shear=(.1,3.)),
            # RandomAffine(p=0.2, degrees=5),
            #RandomResizeCrop(p=1., scale=(0.6, 1.), ratio=(3./4., 4./3.))

            RandomResizeCrop(p=1., scale=(0.3, 0.5), ratio=(3./4., 4./3.)) # ssn
        ]
        if not is_3D:
            transforms.append(lambda x, y: (x, y.squeeze().type(torch.long)))
            return PairedCompose(transforms)
        else:
            transforms.append(CropZDim(size=32, minimum=0, maximum=-1))
            transforms.append(lambda x, y: (x, y.squeeze().type(torch.long)))
            return PairedCompose(transforms)
        
def train_val_test_split(dataset, val_prop, test_prop, seed):
        # I think the sklearn version might be prefereable for determinism and things
        # but that involves fiddling with the dataset implementation I think....
        size = len(dataset)
        test_size = int(test_prop*size) 
        val_size = int(val_prop*size)
        train_size = size - val_size - test_size
        train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
        return train, val, test
    
    
import torch
import torch.nn as nn
from trustworthai.models.uq_models.drop_UNet import normalization_layer
import torch.nn.functional as F
from trustworthai.models.uq_models.initial_variants.HyperMapp3r_deterministic import HyperMapp3r
import torch.distributions as td
from typing import Tuple


def get_conv_func(dims, transpose=False):
    # determine convolution func
        if dims == 2:
            if transpose:
                return nn.ConvTranspose2d
            else:
                return nn.Conv2d
        elif dims == 3:
            if transpose:
                return nn.ConvTranspose3d
            else:
                return nn.Conv3d
        else:
            raise ValueError(f"values of dims of 2 or 3 (2D or 2D conv) are supported only, not {dims}")
            
def get_dropout_func(dims):
    if dims == 2:
        return nn.Dropout2d
    if dims == 3:
        return nn.Dropout3d
    else:
        return nn.Dropout

class ReshapedDistribution(td.Distribution):
    def __init__(self, base_distribution: td.Distribution, new_event_shape: Tuple[int, ...]):
        super().__init__(batch_shape=base_distribution.batch_shape, event_shape=new_event_shape, validate_args=False)
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape
        
        #print("base distribution: ", self.base_distribution)

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints()

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(sample_shape + self.new_shape)

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()
    
    
class MixtureDistribution(td.Distribution):
    def __init__(self, base_distributions, pis):
        super().__init__(batch_shape=base_distributions[0].batch_shape, event_shape=base_distributions[0].event_shape, validate_args=False)
        self.base_distributions = base_distributions
        self.pis = pis
    
    def rsample(self, sample_shape=torch.Size()):
        batch_size = self.pis.shape[0]
        outs = [self.base_distributions[i].rsample(sample_shape) * self.pis[:,i].view(1, batch_size, 1, 1, 1) for i in range(len(self.base_distributions))]
        return torch.stack(outs, dim=0).sum(dim=0)
    
    
class HyperMapp3rSSN2(HyperMapp3r):
    def __init__(self, dims=3,
                 in_channels=3,
                 out_channels=1,
                 encoder_features=[16, 32, 64, 128, 256],
                 decoder_features=[128, 64, 32, 16],
                 softmax=True,
                 up_res_blocks=False,
                 block_params={
                     "dropout_p":0.1,
                     "norm_type":"in", 
                     "dropout_both_layers":False,
                 },
                 ssn_rank = 10,
                 ssn_epsilon=1e-5,
                 ssn_diagonal=False,
                 mixture_size = 5
                ):
        out_multiplier = 2
        super().__init__(dims=dims,
                 in_channels=in_channels,
                 out_channels=decoder_features[-1] * out_multiplier, # last layer of just keeps number of nodes fixed this time
                 encoder_features=encoder_features,
                 decoder_features=decoder_features,
                 softmax=False,
                 up_res_blocks=up_res_blocks,
                 block_params=block_params)
        
        print("WARNING: this model assumes that the input to the model contains the brain mask in the first channel!")
        conv_func = get_conv_func(dims, transpose=False)
        self.ssn_rank = ssn_rank
        self.ssn_diagonal = ssn_diagonal
        self.ssn_epsilon = ssn_epsilon
        self.ssn_num_classes = out_channels
        
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.mean_ls = nn.ModuleList([conv_func(decoder_features[-1]*out_multiplier, out_channels, kernel_size = (1,) *  dims, padding='same') for _ in range(mixture_size)])
        self.log_cov_diag_ls = nn.ModuleList([conv_func(decoder_features[-1]*out_multiplier, out_channels, kernel_size = (1,) * dims, padding='same') for _ in range(mixture_size)])
        self.cov_factor_ls = nn.ModuleList([conv_func(decoder_features[-1]*out_multiplier, out_channels * ssn_rank, kernel_size = (1,) * dims, padding='same') for _ in range(mixture_size)])
        
        self.pi_l = nn.Sequential(nn.Conv2d(decoder_features[-1]*out_multiplier, mixture_size, 7, stride=7), nn.Conv2d(mixture_size, mixture_size, 3, stride=3), nn.AvgPool2d(4))
        self.mixture_size = mixture_size
        
    def forward(self, x):
        logits = self.lrelu(super().forward(x))
        if torch.sum(torch.isnan(logits)) > 0:
            print("NAN 1", torch.sum(torch.isnan(logits)))
        batch_size = logits.shape[0]
        event_shape = (self.ssn_num_classes,) + logits.shape[2:]
        
        means = [mean_l(logits) for mean_l in self.mean_ls]
        means = [mean.view((batch_size, -1)) for mean in means]
        
        cov_diags = [log_cov_diag_l(logits).exp()  for log_cov_diag_l in self.log_cov_diag_ls]
        cov_diags = [cov_diag.view((batch_size, -1)) for cov_diag in cov_diags]
        
        cov_factors = [cov_factor_l(logits) for cov_factor_l in self.cov_factor_ls]
        cov_factors = [cov_factor.view((batch_size, self.ssn_rank, self.ssn_num_classes, -1)) for cov_factor in cov_factors]
        cov_factors = [cov_factor.flatten(2,3) for cov_factor in cov_factors]
        cov_factors = [cov_factor.transpose(1,2) for cov_factor in cov_factors]
        
        pis = self.pi_l(logits)
        pis = pis.view(batch_size, self.mixture_size, -1)
        pis = pis.max(dim=2)[0]
        pis = torch.nn.functional.softmax(pis, dim=1)
        
        # if torch.sum(torch.isnan(mean)) > 0:
        #     print("NAN 2")
        # if torch.sum(torch.isnan(cov_diag)) > 0:
        #     print("NAN 3")
        # if torch.sum(torch.isnan(cov_factor)) > 0:
        #     print("NAN 4")
        
        # covariance tends to blow up to infinity, hence set to 0 outside the ROI
        mask = x[:,1]
        mask = mask.unsqueeze(1).expand((batch_size, self.ssn_num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factors = [(cov_factor * mask.unsqueeze(-1)) for cov_factor in cov_factors]
        cov_diags = [(cov_diag * mask) + self.ssn_epsilon for cov_diag in cov_diags]
        # print([torch.min(cov_diag) for cov_diag in cov_diags])
        # print(torch.min(mask))
        
        # if torch.sum(torch.isnan(mask)) > 0:
        #     print("NAN 5")
        # if torch.sum(torch.isnan(cov_factor)) > 0:
        #     print("NAN 6")
        # for cov_diag in cov_diags:
        #     if torch.sum(torch.isnan(cov_diag)) > 0:
        #         print("NAN 7")
        
        if self.ssn_diagonal:
            base_distributions = [td.Independent(td.Normal(loc=means[i], scale=torch.sqrt(cov_diags[i])), 1) for i in range(self.mixture_size)]
        else:
            try:
                base_distributions = [td.LowRankMultivariateNormal(loc=means[i], cov_factor=cov_factors[i], cov_diag=cov_diags[i]) for i in range(self.mixture_size)]
                #print("using multivariate normal!")
            except Exception as e:
                print("was thrown: ", e)
                print('hmm: Covariance became non invertible using independent normals for this batch!')
                print("cov diag okay: ", [torch.sum(cov_diag <=0) for cov_diag in cov_diags])
                print("sqrt cov diag okay: ", [torch.sum(torch.isnan(torch.sqrt(cov_diag))) for cov_diag in cov_diags])
                print(torch.sqrt(cov_diags[0]))
                print(cov_diags[0])
                
                try:
                    base_distributions = [td.Independent(td.Normal(loc=means[i], scale=torch.sqrt(cov_diags[i])), 1) for i in range(self.mixture_size)]
                except Exception as e:
                    print("second fail: ", e)
                    print(torch.min(torch.sqrt(cov_diags[0]), torch.max(torch.sqrt(cov_diags[0]))))
        
        distributions = [ReshapedDistribution(base_distributions[i], event_shape) for i in range(len(means))]
        distribution = MixtureDistribution(distributions, pis)
        
        shape = (batch_size,) + event_shape
        # print(means[0].shape)
        # print(pis.shape)
        logit_mean_view = [means[i] * pis[:,i].view(batch_size, 1) for i in range(self.mixture_size)]
        # print(logit_mean_view[0].shape)
        logit_mean_view = torch.stack(logit_mean_view, dim=0).sum(dim=0)
        logit_mean_view = logit_mean_view.view(shape).detach()
        # cov_diag_view = cov_diag.view(shape).detach()
        # cov_factor_view = cov_factor.transpose(2,1).view((batch_size, self.ssn_num_classes * self.ssn_rank) + event_shape[1:]).detach()
        
        
        output_dict = {
            'logit_mean':logit_mean_view,
            # 'cov_diag':cov_diag_view,
            # 'cov_factor':cov_factor_view,
            'distribution':distribution,
        }
        
        return output_dict
    
class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        # assert num_samples % 2 == 0
        # samples = dist.rsample((num_samples // 2,))
        # mean = dist.mean.unsqueeze(0)
        # samples = samples - mean
        # return torch.cat([samples, -samples]) + mean
        samples = dist.rsample((num_samples,))
        # mean = dist.mean.unsqueeze(0)
        # samples = samples - mean
        # return torch.cat([samples, -samples]) + mean
        return samples

    def forward(self, result_dict, target, **kwargs):
        logits = result_dict['logit_mean']
        distribution = result_dict['distribution']
        
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        assert num_classes >= 2  # not implemented for binary case with implied background
        # logit_sample = distribution.rsample((self.num_mc_samples,))
        logit_sample = self.fixed_re_parametrization_trick(distribution, self.num_mc_samples)
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)

        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))

        log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view((self.num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples))
        loss = -loglikelihood
        return loss
    
def fixed_re_parametrization_trick(dist, num_samples):
        # assert num_samples % 2 == 0
        # samples = dist.rsample((num_samples // 2,))
        # mean = dist.mean.unsqueeze(0)
        # samples = samples - mean
        # return torch.cat([samples, -samples]) + mean
        samples = dist.rsample((num_samples,))
        # mean = dist.mean.unsqueeze(0)
        # samples = samples - mean
        # return torch.cat([samples, -samples]) + mean
        return samples


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
    
class SsnDiceMetricWrapper(DiceLossMetric):

    def update(self, preds_dict, target: torch.Tensor):
        super().update(preds_dict['logit_mean'], target)

    def compute(self):
        return super().compute()
    
ssn_diceloss = SsnNetworkSampleLossWrapper(dice_loss)# SsnNetworkMeanLossWrapper(dice_loss)
mc_loss = StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=10)
    
class StandardLitModelWrapper(pl.LightningModule):
        def __init__(self, model, loss=F.cross_entropy, logging_metric=None, optimizer_params={"lr":1e-3}, lr_scheduler_params={"step_size":30, "gamma":0.1}, is_uq_model=False,
                    optimizer_constructor=None, lr_scheduler_constructor=None):
            super().__init__()
            self.model = model
            self.loss = loss
            self.logging_metric_train = logging_metric()
            self.logging_metric_val = logging_metric()
            self.optim_params = optimizer_params
            self.lr_scheduler_params = lr_scheduler_params
            self.is_uq_model = False
            self.optimizer_constructor = optimizer_constructor
            self.lr_scheduler_constructor = lr_scheduler_constructor


        def forward(self, x, **kwargs):
            return self.model(x, **kwargs)

        def configure_optimizers(self):
            # optimizer and schedulers go in the configure optimizers hook
            if self.optimizer_constructor:
                optimizer = self.optimizer_constructor(self.parameters(), **self.optim_params)
            else:
                optimizer = torch.optim.Adam(self.parameters(), **self.optim_params)

            if self.lr_scheduler_constructor:
                lr_scheduler = self.lr_scheduler_constructor(optimizer, **self.lr_scheduler_params)
            else:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_scheduler_params)

            return [optimizer], [lr_scheduler]

        def training_step(self, batch, batch_idx):
            """
            lightning automates the training loop, 
            does epoch, back_tracking, optimizers and schedulers,
            and metric reduction.
            we just define how we want to process a single batch. 
            we can optionally pass optimizer_idx if we want to define multiple optimizers within the configure_optimizers
            hook, and I presume we can add our own parameters also to functions?
            """

            if self.is_uq_model:
                self.model.set_applyfunc(True)

            X, y = batch
            y_hat = self(X)
            loss = self.loss(y_hat, y)

            # metrics 
            if self.logging_metric_train:
                self.logging_metric_train(y_hat, y)
                self.log(f"train_metric", self.logging_metric_train, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train_loss", loss)

            return loss

    #     def training_epoch_end(self, outs):
    #         self.log('train_metric_epoch', self.logging_metric_train.compute())

    #     def validation_epoch_end(self, outs):
    #         self.log('val_metric_epoch', self.logging_metric_val.compute())

        def validation_step(self, batch, batch_idx):
            """
            note: call trainer.validate() automatically loads the best checkpoint if checkpointing was enabled during fitting
            well yes I want to enable checkpointing but will deal with that later.
            also it does stuff like model.eval() and torch.no_grad() automatically which is nice.
            I will need a custom eval thing to do my dropout estimation but can solve that later too.
            """
            if self.is_uq_model:
                self.model.set_applyfunc(False)

            X, y = batch
            y_hat = self(X)
            val_loss = self.loss(y_hat, y)

            if self.logging_metric_val:
                self.logging_metric_val(y_hat, y)
                self.log(f"val_metric", self.logging_metric_val, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val_loss", val_loss)

        def test_step(self, batch, batch_idx):
            """
            we would need to directly call this function using the trainer
            """

            if self.is_uq_model:
                self.model.set_applyfunc(False)

            X, y = batch
            y_hat = self(X)
            test_loss = self.loss(y_hat, y)
            self.log("test_loss", test_loss)

        def predict_step(self, batch, batch_idx):
            """
            just for making predictions as opposed to collecting metrics etc
            note to use this, we just call .predict(dataloader) and it then automates the look
            these functions are for a single batch. Nice.
            """
            X, y = batch
            pred = self(X)
            return pred

def main(args):
    dice_factor = args.dice_factor
    
    def double_loss(outs, target):
        return ssn_diceloss(outs, target) * dice_factor + mc_loss(outs, target) * 0.01
    
    ckpt_dir = args.ckpt_dir
    is3D = False
    root_dir = "/disk/scratch/s2208943/ipdis/preprep/out_data/collated/"
    #root_dir = "/media/benp/NVMEspare/datasets/preprocessing_attempts/local_results/collated/"
    wmh_dir = root_dir + "WMH_challenge_dataset/"
    ed_dir = root_dir + "EdData/"

    domains = [
                ed_dir + d for d in ["domainA", "domainB", "domainC", "domainD"]
              ]

    test_proportion = 0.1
    validation_proportion = 0.2
    seed = 3407

    # load datasets
    # this step is quite slow, all the data is being loaded into memory
    if is3D:
        datasets_domains = [MRISegmentation3DDataset(root_dir, domain, transforms=get_transforms(is_3D=True)) for domain in domains]
    else:
        datasets_domains = [MRISegmentation2DDataset(root_dir, domain, transforms=get_transforms(is_3D=False)) for domain in domains]

    # split into train, val test datasets
    datasets = [train_val_test_split(dataset, validation_proportion, test_proportion, seed) for dataset in datasets_domains]

    # concat the train val test datsets
    train_dataset = ConcatDataset([ds[0] for ds in datasets])
    val_dataset = ConcatDataset([ds[1] for ds in datasets])
    test_dataset = ConcatDataset([ds[2] for ds in datasets])

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size = 16, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model_raw = HyperMapp3rSSN2(dims=2,
                 in_channels=3,
                 out_channels=2,
                 encoder_features=[16, 32, 64, 128, 256],
                 decoder_features=[128, 64, 32, 16],
                 softmax=False,
                 up_res_blocks=False,
                 block_params={
                     "dropout_p":0.1,
                     "norm_type":"in", 
                     "dropout_both_layers":False,
                 },
                 ssn_rank = 10,
                 ssn_epsilon=1e-5,
                 ssn_diagonal=False)

    loss = double_loss

    optimizer_params={"lr":1e-4}
    optimizer = torch.optim.Adam
    lr_scheduler_params={"milestones":[50,80], "gamma":0.5}
    lr_scheduler_constructor = torch.optim.lr_scheduler.MultiStepLR

    model = StandardLitModelWrapper(model_raw, loss, 
                                    logging_metric=lambda : None,
                                    optimizer_params=optimizer_params,
                                    lr_scheduler_params=lr_scheduler_params,
                                    is_uq_model=False,
                                    optimizer_constructor=optimizer,
                                    lr_scheduler_constructor=lr_scheduler_constructor
                                   )

    accelerator="gpu"
    devices=1
    max_epochs=100
    precision = 32

    rootdir = "/disk/scratch/s2208943/results/final_models/"
    final_dir = rootdir + ckpt_dir
    checkpoint_callback = ModelCheckpoint(final_dir, save_top_k=2, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=15, verbose="False", mode="min", check_finite=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        default_root_dir=final_dir
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)