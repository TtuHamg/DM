import torch as th
import math
import numpy as np

# --------------------unet.py------------------- #


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


# --------------------gaussian_diffusion.py------------------- #


def extract_into_tensor(arr, timestep, broadcast_shape):
    coe = th.from_numpy(arr).to(device=timestep.device)[timestep].float()
    while len(coe.shape) < len(broadcast_shape):
        coe = coe[..., None]
    return coe.expand(broadcast_shape)  # other place need to aseert shape consistent


def create_beta_schedule(schedule_name, num_diffusion_time_step):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_time_step
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_time_step, dtype=np.float64
        )
    elif schedule_name == "cosine":
        pass


def scale_time_step(t, num_time_step):
    """according to the num_time_step to scale t

    Args:
        t (float list): time step
        num_time_step (int): total number step

    Returns:
        float: scaled time step
    """
    return t.float() * (1000.0 / num_time_step)
