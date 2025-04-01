import torch.optim as top
from typing import Iterable
import re

class OptimizerConfigurator():
    def __init__(self, optim_string, lr_strings=None):
        optim_constructor, optim_params = self.extract_config(top, optim_string)
        self.optim_constructor = optim_constructor
        self.optim_params = optim_params
        
        if lr_strings:
            if isinstance(lr_strings, str):
                lr_strings = [lr_strings]
            self.lr_configs = [self.extract_config(top.lr_scheduler, lrs) for lrs in lr_strings] 
        else:
            self.lr_configs = None
            
    def __call__(self, parameters):
        optimizer = self.optim_constructor(parameters, **self.optim_params)
        
        if self.lr_configs:
            lr_schedulers = [
                lr_constructor(optimizer, **lr_params)
                for (lr_constructor, lr_params) in self.lr_configs
            ]
            if len(lr_schedulers) == 1:
                lr_scheduler = lr_schedulers[0]
            else:
                lr_scheduler = top.ChainedScheduler(lr_schedulers)
                
            configuration =  {
                "optimizer":optimizer,
                "lr_scheduler": {
                    "scheduler":lr_scheduler,
                    # 'epoch' updates the scheduler on epoch end, 'step'
                    # updates it after a optimizer update.
                    "interval":"epoch",
                    # rate after every epoch/step.
                    "frequency":1,

                }
            }
            return configuration
            
        return optimizer
    
    def load_class_constructor(self, module, cname):
        try:
            return getattr(module, cname)
        except:
            raise ValueError(f"could not find {cname} in {module}")
    

    def extract_config(self, module, string):
        string = string.replace("\n", "")
        parts = string.split(" ")
        cname = parts[0]
        constructor = self.load_class_constructor(module, cname)
        params = [p.split(":") for p in parts[1:]]
        params = {p[0]:self.type_convert_param(p[1]) for p in params}

        return constructor, params
    
    def type_convert_param(self, param_string):
        assert param_string != ""
        """
        convert the param string to a float, int, bool or string as approporiate
        """
        
        # check that it is a number, 123, 123.401, 1.2e-40 1e6 1e-6 .8 all allowed
        if re.match(r'^-?(\d+)?(?:\.\d+)?(e-?[1-9]\d?)?$', param_string) is not None:
            # check if it is an int or not:
            if "." in param_string or "e" in param_string:
                return float(param_string)
            else:
                return int(param_string)
            
        # check if it is a boolean
        elif param_string.lower() == "true":
             return True
        elif param_string.lower() == "false":
            return False
        # it should be a string
        else:
            return param_string
    
    