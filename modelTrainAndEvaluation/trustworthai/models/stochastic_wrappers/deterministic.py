"""
deterministic model is just a do nothing wrapper
"""
from trustworthai.models.uq_model import UncertaintyQuantificationModel

class Deterministic(UncertaintyQuantificationModel):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return self.base_model(x)
    
    def mean(self, x, temperature=1):
        return self(x) / temperature
    
    def _samples_from_dist(self, dist, num_samples, rsample=True, symmetric=True):
        raise ValueError("not implemented for deterministic model!")
    
    def mean_and_sample(self, x, num_samples, rsample=True, temperature=1):
        raise ValueError("not implemented for deterministic model!")