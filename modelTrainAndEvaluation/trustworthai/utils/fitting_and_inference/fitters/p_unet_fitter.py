import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from trustworthai.models.uq_model import UncertaintyQuantificationModel

class PUNetLitModelWrapper(pl.LightningModule):
    def __init__(self, model, loss=F.cross_entropy, use_prior_for_dice=False, logging_metric=None, optimizer_params={"lr":1e-3}, lr_scheduler_params={"step_size":30, "gamma":0.1}, 
                optimizer_constructor=None, lr_scheduler_constructor=None, val_loss=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.logging_metric_train = logging_metric()
        self.logging_metric_val = logging_metric()
        self.optim_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.optimizer_constructor = optimizer_constructor
        self.lr_scheduler_constructor = lr_scheduler_constructor
        self.use_prior_for_dice = use_prior_for_dice
        self.extra_val_loss=val_loss

        
    def forward(self, x, y, **kwargs):
        return self.model(x, y, **kwargs)
    
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
        
        X, y = batch
        self(X, y.unsqueeze(1), training=True)
        # dice: This term is not part of the ELBO, so we are free to sample from
        # the prior network since this is used at test-time.
        #y_hat = self.model.sample(testing=False) # or I can use reconstruct to get samples from the posterior...
        if self.use_prior_for_dice:
            #print("in use!")
            y_hat = self.model.sample(testing=False)
            y_hat_mean = self.model.sample(use_prior_mean=True)
        else:
            y_hat = self.model.reconstruct(
                use_posterior_mean=False, calculate_posterior=True)
            y_hat_mean = self.model.reconstruct(use_posterior_mean=True)
                
        loss = self.loss(self.model, y_hat, y_hat_mean, y)
        
        # metrics 
        if self.logging_metric_train:
            self.logging_metric_train(y_hat, y)
            self.log(f"train_metric", self.logging_metric_train, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        note: call trainer.validate() automatically loads the best checkpoint if checkpointing was enabled during fitting
        well yes I want to enable checkpointing but will deal with that later.
        also it does stuff like model.eval() and torch.no_grad() automatically which is nice.
        I will need a custom eval thing to do my dropout estimation but can solve that later too.
        """
        
        X, y = batch
        self(X, y.unsqueeze(1), training=True)
        if self.use_prior_for_dice:
            #print("in use!")
            y_hat = self.model.sample(testing=False)
            y_hat_mean = self.model.sample(use_prior_mean=True)
        else:
            y_hat = self.model.reconstruct(
                use_posterior_mean=False, calculate_posterior=True)
            y_hat_mean = self.model.reconstruct(use_posterior_mean=True)
            
        if self.extra_val_loss != None:
            val_loss = self.extra_val_loss(self.model, y_hat, y_hat_mean, y)
        else:
            val_loss = self.loss(self.model, y_hat, y_hat_mean, y)
        
        if self.logging_metric_val:
            self.logging_metric_val(y_hat, y)
            self.log(f"val_metric", self.logging_metric_val, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss)