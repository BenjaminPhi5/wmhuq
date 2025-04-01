import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from trustworthai.models.uq_model import UncertaintyQuantificationModel


class StandardLitModelWrapper(pl.LightningModule, UncertaintyQuantificationModel):
        def __init__(self, model, loss=F.cross_entropy, logging_metric=None, optimizer_params={"lr":1e-3}, lr_scheduler_params={"step_size":30, "gamma":0.1},
                    optimizer_constructor=None, lr_scheduler_constructor=None, do_log=True, val_loss=None):
            super().__init__()
            self.model = model
            self.loss = loss
            self.logging_metric_train = logging_metric()
            self.logging_metric_val = logging_metric()
            self.optim_params = optimizer_params
            self.lr_scheduler_params = lr_scheduler_params
            self.optimizer_constructor = optimizer_constructor
            self.lr_scheduler_constructor = lr_scheduler_constructor
            self.do_log = do_log
            self.extra_val_loss = val_loss


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
            X, y = batch
            y_hat = self(X)
            loss = self.loss(y_hat, y)

            # metrics 
            if self.do_log:
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
            y_hat = self(X)
            
            if self.extra_val_loss != None:
                val_loss = self.extra_val_loss(y_hat, y)
            else:
                val_loss = self.loss(y_hat, y)

            if self.do_log:
                if self.logging_metric_val:
                    self.logging_metric_val(y_hat, y)
                    self.log(f"val_metric", self.logging_metric_val, on_step=True, on_epoch=True, prog_bar=True)
                self.log("val_loss", val_loss)

        def test_step(self, batch, batch_idx):
            """
            we would need to directly call this function using the trainer
            """

            X, y = batch
            y_hat = self(X)
            test_loss = self.loss(y_hat, y)
            
            if self.do_log:
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

        def mean(self, *args, **kwargs):
            return self.model.mean(*args, **kwargs)

        def mean_and_sample(self, *args, **kwargs):
            return self.model.mean_and_sample(*args, **kwargs)

        def sample_over_3Ddataset(self, *args, **kwargs):
            return self.model.sample_over_3Ddataset(*args, **kwargs)