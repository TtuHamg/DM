import torch as th 

def zero_module(module):
    """set module's paramters to zero and return it.

    Args:
        module (nn.Module): 

    Returns:
        nn.Module: parameters seted to zero
    """
    for p in module.parameters():
        p.detach().zero_()#
    return module