from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from typing import Union
import pydoc

def get_network_from_plans(
    arch_class_name,
    arch_kwargs,
    arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
    input_channels=3,
    output_channels=2,
    allow_init=True,
    deep_supervision: Union[bool, None] = None
):
    
    arch_class_name = arch_class_name.split(".")[-1]
    if arch_class_name == "ResidualEncoderUNet":
        constructor = ResidualEncoderUNet
    elif arch_class_name == 'PlainConvUNet':
        constructor = PlainConvUNet
    else:
        raise ValueError("architecture constructor unknown")
     
    architecture_kwargs = dict(**arch_kwargs)
    if arch_kwargs_req_import is not None:
        for ri in arch_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
    
    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = constructor(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network
