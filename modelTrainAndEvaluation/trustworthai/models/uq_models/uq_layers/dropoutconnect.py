import torch
import torch.nn.functional as F
from torch.nn import Parameter
from enum import Enum, auto
from trustworthai.models.uq_models.uq_model import UQModel

# ======================================================================================
# Custom flag deciding when to applyfunc our custom dropout 
# (we want to keep our dropout/dropconnect turned on during uncertainty quantification)
# ======================================================================================
class ApplyFuncEffectFlag(Enum):
    TRAINING = auto()
    UNCERT_QUANT = auto()
    PREDICTION = auto()
    
class BoxedapplyfuncFlag:
    def __init__(self, aef:ApplyFuncEffectFlag):
        self.aef = aef
    
    def get_training_state(self):
        if self.aef is ApplyFuncEffectFlag.PREDICTION:
            return False
        elif self.aef is ApplyFuncEffectFlag.UNCERT_QUANT:
            return True
        elif self.aef is ApplyFuncEffectFlag.TRAINING:
            return True
        
    def set_training_state(self, aef:ApplyFuncEffectFlag):
        self.aef = aef
        

# ======================================================================================
# Custom Dropout Wrapper
# (the same as normal dropout but we have the option to keep the dropout effect
# on (applyfunc=True) during inference also for uncertainty quantification
# and then turn it off making the actual prediction.
# ======================================================================================
class UQDropout(UQModel):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        
    def forward(self, x):
        return F.dropout(x, self.p, self.applyfunc, self.inplace)

class UQDropout2d(UQDropout):
    def forward(self, x):
        return F.dropout2d(x, self.p, self.applyfunc, self.inplace)

class UQDropout3d(UQDropout):
    def forward(self, x):
        return F.dropout3d(x, self.p, self.applyfunc, self.inplace)

# ======================================================================================
# Gaussian Dropout helper functions
# ======================================================================================
def _gaussian_dropout(x, mean, alpha, applyfunc, inplace):
    if applyfunc and alpha > 0.:
        mean = torch.ones_like(x, dtype=x.dtype, device=x.device)
        gaussian_noise = torch.normal(mean, alpha)
        x = x.mul_(gaussian_noise) if inplace else x.mul(gaussian_noise)
    return x

def _gaussian_feature_dropout(x, mean, alpha, applyfunc, inplace, x_dims):
    if applyfunc and alpha > 0.:
        shape = x.shape
        dropout_shape = torch.ones(x_dims, dtype=int)
        dropout_shape[0] = shape[0]
        dropout_shape[1] = shape[1]
        mean = torch.ones(torch.Size(dropout_shape.tolist()), dtype=x.dtype, device=x.device)
        gaussian_noise = torch.normal(mean, alpha)
        x = x.mul_(gaussian_noise) if inplace else x.mul(gaussian_noise)
    return x
        
def _gaussian_dropout2d(x, mean=1., alpha = 0.5, applyfunc = True, inplace = False):
    # inspired by the torch.nn.functional versions of standard dropout
    inp_dim = x.dim()
    is_batched = inp_dim == 4
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = _gaussian_feature_dropout(x, mean, alpha, applyfunc, inplace, x_dims=4)

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)

    return result


def _gaussian_dropout3d(x, mean=1., alpha = 0.5, applyfunc = True, inplace = False):
    inp_dim = x.dim()

    is_batched = inp_dim == 5
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = _gaussian_feature_dropout(x, mean, alpha, applyfunc, inplace, x_dims=5)

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)
    return result

# ======================================================================================
# Gaussian Dropout classes
# ======================================================================================

class UQGaussianDropout(UQModel):
    def __init__(self, mean, p=0.0, inplace=False):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.alpha = p / (1-p)
        self.mean = mean
        self.inplace = inplace
        
    def forward(self, x):
        return _gaussian_dropout(x, self.mean, self.alpha, applyfunc=self.applyfunc, inplace=self.inplace)
    
class UQGaussianDropout2d(UQGaussianDropout):
    def forward(self, x):
        return _gaussian_dropout2d(x, self.mean, self.alpha, applyfunc=self.applyfunc, inplace=self.inplace)
    
class UQGaussianDropout3d(UQGaussianDropout):
    def forward(self, x):
        return _gaussian_dropout3d(x, self.mean, self.alpha, applyfunc=self.applyfunc, inplace=self.inplace)
    
    
# ======================================================================================
# drop connect helper functions
# ======================================================================================
    
def _weight_dropconnect(module, weights, dropout, dropout_func):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(applyfunc, *args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            # print(args)
            # print(kwargs)
            # print((raw_w, dropout, applyfunc))
            w = dropout_func(raw_w, p=dropout, training=applyfunc)
            setattr(module, name_w, Parameter(w))

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)
    
def _weight_gaussianconnect(module, weights, mean, alpha):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(applyfunc, *args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            # note, we don't do in-place on the model weights!!
            w = _gaussian_dropout(raw_w, mean=mean, alpha=alpha, applyfunc=applyfunc, inplace=False)
            setattr(module, name_w, Parameter(w))

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)
    
def _weight_gaussianconnect_multidim(module, weights, mean, alpha, x_dims):
    """
    xdims = 4 for 2D, and 5 for 3D (in planes, out_planes, c, ...)
    """
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(applyfunc, *args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            # note, we don't do in-place on the model weights!!
            w = _gaussian_feature_dropout(raw_w, mean=mean, alpha=alpha, applyfunc=applyfunc, inplace=False, x_dims=x_dims)
            setattr(module, name_w, Parameter(w))

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)

    
# ======================================================================================
# Drop Connect classes
# ======================================================================================
    
class UQDropConnect(UQModel):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    Adapted from 
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    license:
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`UQModel`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to applyfunc a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> from torchnlp.nn import WeightDrop
        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights=None, p=0.0):
        super().__init__()
        self.applyfunc = True
        self.p = p
        if weights == None:
            weights = ['weight']
        self._setup(module, weights)
        self.forward_module = module.forward
        self.inner_layer = module
        
    def _setup(self, module, weights):
        _weight_dropconnect(module, weights, self.p, F.dropout)
        
    def forward(self, *args, **kwargs):
        return self.forward_module(self.applyfunc, *args, **kwargs)
        
class UQDropConnect2d(UQDropConnect):
    def _setup(self, module, weights):
        _weight_dropconnect(module, weights, self.p, F.dropout2d)
        
class UQDropConnect3d(UQDropConnect):
    def _setup(self, module, weights):
        _weight_dropconnect(module, weights, self.p, F.dropout3d)
        
        
# ======================================================================================
# Gaussian Drop Connect classes
# ======================================================================================
class UQGaussianConnect(UQModel):
    def __init__(self, module, weights, mean, p):
        super().__init__()
        alpha = p / (1-p)
        if weights == None:
            weights = ['weight']
        _weight_gaussianconnect(module, weights, mean, alpha)
        self.forward_module = module.forward
        self.inner_layer = module
        
    def forward(self, *args, **kwargs):
        return self.forward_module(self.applyfunc, *args, **kwargs)
        
class UQGaussianConnect2d(UQModel):
    def __init__(self, module, weights, mean, p):
        super().__init__()
        alpha = p / (1-p)
        if weights == None:
            weights = ['weight']
        _weight_gaussianconnect_multidim(module, weights, mean, alpha, x_dims=4)
        self.forward_module = module.forward
        self.inner_layer = module
        
    def forward(self, *args, **kwargs):
        return self.forward_module(self.applyfunc, *args, **kwargs)
        
class UQGaussianConnect3d(UQModel):
    def __init__(self, module, weights, mean, p):
        super().__init__()
        alpha = p / (1-p)
        if weights == None:
            weights = ['weight']
        _weight_gaussianconnect_multidim(module, weights, mean, alpha, x_dims=5)
        self.forward_module = module.forward
        self.inner_layer = module
        
    def forward(self, *args, **kwargs):
        return self.forward_module(self.applyfunc, *args, **kwargs)