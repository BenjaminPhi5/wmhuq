from wmhuq.models import HyperMapp3r
from wmhuq.models import SSN
from wmhuq.models import StandardLitModelWrapper
import os

SSN_PRE_HEAD_LAYERS = 32
DROPOUT_P = 0
SSN_RANK = 25
SSN_EPSILON = 1e-05
IN_CHANNELS = 3
OUT_CHANNELS = 2
DIMS = 2
ENCODER_SIZES = [16,32,64,128,256]

def load_ssn(device):
    # for SSN and hypermapp3r, for now we use the number of first layer
    # channels as the number of out channels as well, prior to the ssn layer
    # which is of size 32 so maybe we should use 32?
    base_model = HyperMapp3r(
        dropout_p=DROPOUT_P,
        encoder_sizes=ENCODER_SIZES,
        inchannels=IN_CHANNELS,
        outchannels=SSN_PRE_HEAD_LAYERS
    )
    
    # wrap the base model in the SSN head
    model_raw = SSN(
        base_model=base_model,
        rank=SSN_RANK,
        diagonal= (SSN_RANK == 1),
        epsilon=SSN_EPSILON,
        intermediate_channels=SSN_PRE_HEAD_LAYERS,
        out_channels=OUT_CHANNELS,
        dims=DIMS,
    ).to(device)
    
    return model_raw

def load_model_weights(model, ckpt_folder, map_location="cpu"):
    with open(os.path.join(ckpt_folder, "best_ckpt.txt"), "r") as f:
        ckpt_file = os.path.join(ckpt_folder, f.readlines()[0][:-1].split("/")[-1])
    
    model = StandardLitModelWrapper.load_from_checkpoint(ckpt_file, model=model, map_location=map_location, loss=None, logging_metric=lambda : None)
    return model
