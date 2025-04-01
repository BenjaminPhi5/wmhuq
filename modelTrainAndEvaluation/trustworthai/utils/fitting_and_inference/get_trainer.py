from trustworthai.utils.fitting_and_inference.get_scratch_dir import scratch_dir
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
import pytorch_lightning as pl

accelerator="gpu"
devices=1
precision = 32

def get_trainer(max_epochs, results_dir, early_stop_patience, store_ckpts=True):
    rootdir = os.path.join(scratch_dir(), results_dir)
    
    checkpoint_callback = ModelCheckpoint(results_dir, save_top_k=2, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=early_stop_patience, verbose="False", mode="min", check_finite=True)
    callbacks = [early_stop_callback]
    if store_ckpts:
        callbacks.append(checkpoint_callback)
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        default_root_dir=results_dir
    )
    
    return trainer

