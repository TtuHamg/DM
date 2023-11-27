import torch as th
import math


def zero_module(module):
    """set module's paramters to zero and return it.

    Args:
        module (nn.Module):

    Returns:
        nn.Module: parameters seted to zero
    """
    for p in module.parameters():
        p.detach().zero_()  #
    return module


def timestep_embedding(timestep, dim, max_period=10000):
    half = dim // 2
    pos_embed = th.zeros(len(timestep), dim)
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timestep.device)
    args = freqs[None] * timestep[:, None]
    if dim % 2:
        pos_embed[:, 0::2] = th.cat(
            [th.sin(args), th.zeros_like(pos_embed[:, :1])], dim=-1
        )
    else:
        pos_embed[:, 0::2] = th.sin(args)
    pos_embed[:, 1::2] = th.cos(args)

    return pos_embed
