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


def calculate_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be tensor."

    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def mean_flat(kl):
    return kl.mean(dim=list(range(1, len(kl.shape))))


def approx_standard_normal_cdf(x):
    """a fast approximation of the cumulative distribution function(CDF) of the standard model.
    In fact, it's GELU(gaussian error linear unit) implementation in pytorch

    Args:
        x (tensor): standard gaussian

    Returns:
        tensor: the approximation of the standard normal cdf.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """By taking the difference of the cumulative distribution functions of continuous distributions,
    one can simulate discrete distributions

    Args:
        x (tensor): the target image which is uint8 values, rescaled to range [-1,1]
        means (tensor): gaussian mean tensor
        log_scales (tensor): gaussian log std tensor
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_std = th.exp(log_scales)
    plus_in = inv_std * (centered_x + 1.0 / 255.0)
    min_in = inv_std * (centered_x - 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    cdf_min = approx_standard_normal_cdf(min_in)

    # keep stability
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min

    # when x is larger than 0.999, prefer to use log_one_minus_cdf_min to present zero
    # when x is smaller than -0.999, prefer to use log_cdf_plus to ensure stability.
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
