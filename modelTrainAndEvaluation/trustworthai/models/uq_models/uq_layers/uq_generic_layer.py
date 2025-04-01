from torch import nn
from trustworthai.models.uq_models.uq_model import UQModel

class UQLayerWrapper(UQModel):
    def __init__(self, layer: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
    
    def forward(self, *args, **kwargs):
        if self.layer.training != self.applyfunc:
            self.layer.train(self.applyfunc)
        return self.layer(*args, **kwargs)
    
    
class UQLayerWrapperFromConstructable(UQModel):
    def __init__(self, layer_constructable, *args, **kwargs):
        super().__init__()
        self.layer = layer_constructable(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        if self.layer.training != self.applyfunc:
            self.layer.train(self.applyfunc)
        return self.layer(*args, **kwargs)