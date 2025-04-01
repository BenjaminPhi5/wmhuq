from pytorch_lightning.callbacks import Callback
import torch

class LogParamsGradientsCallback(Callback):
    def __init__(self, log_freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_freq=log_freq
        print(self.log_freq)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step % self.log_freq != 0 or pl_module.global_step == 0:
            return 
        # print(pl_module.global_step)
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # print("grad")
                # print(param.grad)
                # print("end of grad")
                
                # just set gradient to 400 if it goes nan. 400 seems to be bigger than it ever gets when im using gradient clipping.
                grad = param.grad.clone()
                param = param.clone()
                grad[torch.isnan(grad)] = 400
                param[torch.isnan(param)] = 400
                
                trainer.logger.experiment.add_histogram(f"grad/{name}", grad, pl_module.global_step)
                trainer.logger.experiment.add_histogram(f"param/{name}", param, pl_module.global_step)
