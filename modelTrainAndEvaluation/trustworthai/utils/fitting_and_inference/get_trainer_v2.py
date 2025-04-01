from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler, StochasticWeightAveraging
from trustworthai.utils.fitting_and_inference.param_and_gradient_logger import LogParamsGradientsCallback
import os
import pytorch_lightning as pl
from trustworthai.utils.fitting_and_inference.get_scratch_dir import scratch_dir

def get_trainer(max_epochs, results_dir, early_stop_patience=50, use_early_stopping=True, early_stop_on_train=False, accelerator="gpu", devices=1, precision="16-mixed", min_delta=0.0, accumulate_grad_batches=1, scheduled_accumuate_gradients=False, do_gradient_clip=False, gradient_clip_val=1, do_stochastic_weight_averaging=False, track_gradients=False, gradient_log_freq=10):

    rootdir = os.path.join(scratch_dir(), results_dir)
    
    checkpoint_callback = ModelCheckpoint(rootdir, save_top_k=2, monitor="val_loss")

    callbacks = [checkpoint_callback]
    if use_early_stopping:

        if early_stop_on_train:
            early_stop_callback = EarlyStoppingOnTraining(monitor="train_loss", min_delta=0.01, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
        else:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
        callbacks.append(early_stop_callback)
        
    if scheduled_accumuate_gradients:
        # till the 5th epoch, accumulate every 8 batches
        # from 5th to 9th epoch it will accumulate every 4 batches
        # after that no accumulation will happen
        accumulator = GradientAccumulationScheduler(scheduling={0:8, 4:4, 8:1})
        callbacks.append(accumulator)
        
    if not do_gradient_clip:
        gradient_clip_val=None
        
    if do_stochastic_weight_averaging:
        callbacks.append(
            StochasticWeightAveraging(swa_lrs=1e-2)
        )
        
    if track_gradients:
        callbacks.append(
            LogParamsGradientsCallback(gradient_log_freq)
        )

    print("clipping using norm algorithm")
    return pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        default_root_dir=rootdir,
        #fast_dev_run=7,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_algorithm="norm",
        gradient_clip_val=gradient_clip_val,
    )
