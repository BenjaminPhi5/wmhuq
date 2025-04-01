import torch.nn as nn

"""
just a simple flag that allows me to manually
control the training flag of specific layers in a model
without it effecting other layers (the UQ layers do this).

Any UQ layer and UQ Model should inherit this and then for any submodules
the flag will be set.
"""

class UQModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applyfunc = True
        
    def _set_applyfuncflag(self, a: bool):
        self.applyfunc = a
        
    def set_applyfunc(self, a: bool):
        """
        sets the internal flag and propagates the
        result to submodules.
        """
        assert type(a) == bool
        if self.applyfunc != a: # for efficiency
            self.applyfunc = a
            for module in self.modules():
                # the module function finds all submodules, including modules of child modules etc
                # the .children() method wont work if a dq model is contained within a sequential!
                if isinstance(module, UQModel):
                    module._set_applyfuncflag(a)
        
    def get_applyfunc(self):
        return self.applyfunc
    
