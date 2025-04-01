import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardLitModelWrapper(pl.LightningModule):
        def __init__(self, model, loss, optimizer_configurator, logging_metric=None, do_log=True, check_finiteness_of_data=False):
            super().__init__()
            self.model = model
            self.loss = loss
            self.logging_metric_train = None if logging_metric == None else logging_metric()
            self.logging_metric_val = None if logging_metric == None else logging_metric()
            self.do_log = do_log
            self.optimizer_configurator = optimizer_configurator
            self.check_finiteness_of_data = check_finiteness_of_data
            if self.check_finiteness_of_data:
                print("checking if X, y, prediction and loss are finite during training and validation")
            

        def forward(self, x, **kwargs):
            return self.model(x, **kwargs)

        def configure_optimizers(self):
            return self.optimizer_configurator(self.parameters())

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
            
            if self.check_finiteness_of_data:
                assert torch.isfinite(X).all().item()
                assert torch.isfinite(y).all().item()
                assert torch.isfinite(y_hat).all().item()
                assert torch.isfinite(loss).all().item()

            # metrics 
            if self.do_log:
                if self.logging_metric_train:
                    value = self.logging_metric_train(y_hat, y)
                    self.log(f"train_metric", value, on_step=True, on_epoch=False, prog_bar=True)
                self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

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
            
            val_loss = self.loss(y_hat, y)
            
            if self.check_finiteness_of_data:
                assert torch.isfinite(X).all().item()
                assert torch.isfinite(y).all().item()
                assert torch.isfinite(y_hat).all().item()
                assert torch.isfinite(val_loss).all().item()

            if self.do_log:
                if self.logging_metric_val:
                    value = self.logging_metric_val(y_hat, y)
                    self.log(f"val_metric", value, on_step=True, on_epoch=False, prog_bar=True)
                self.log("val_loss", val_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        def test_step(self, batch, batch_idx):
            """
            we would need to directly call this function using the trainer
            """

            X, y = batch
            y_hat = self(X)
            test_loss = self.loss(y_hat, y)
            
            if self.do_log:
                self.log("test_loss", test_loss.item())

        def predict_step(self, batch, batch_idx):
            """
            just for making predictions as opposed to collecting metrics etc
            note to use this, we just call .predict(dataloader) and it then automates the look
            these functions are for a single batch. Nice.
            """
            X, y = batch
            pred = self(X)
            return pred