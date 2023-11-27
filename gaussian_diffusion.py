import torch as th
import numpy as np
import math
import einops
import utils as tools
import enum

from unet import Unet

"""
    notes: q means true process of diffusion and denoising
           p means process related to model
"""


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        rescale_time_step=False,
    ):
        self.rescale_time_step = rescale_time_step

        self.num_time_step = int(betas.shape[0])
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # q_posterior: q(x_{t-1} | q_{x}, q_{0})
        self.posterior_mean_coef_x0 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef_xt = (
            np.sqrt(self.alphas)
            * (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            (1 - self.alphas_cumprod_prev) * betas / (1 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """get mean and variance about p(x_t | x_0)"""
        mean = tools.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        variance = tools.extract_into_tensor(
            1.0 - self.alphas_cumprod, t, x_start.shape
        )
        assert mean.shape[0] == variance.shape[0] == x_start.shape[0]

        return mean, variance

    def q_sample(self, x_start, t, noise=None):
        """sample from q(x_t | x_0)

        Args:
            x_start (tensor): the input of DM
            t (float/list): add noise until t step
            noise (tensor, optional): gaussian noise. Defaults to None.

        Returns:
            tensor: x_t
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
            tools.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + tools.extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """get mean and variance about posterior p(x_{t-1} | x_{t}, x_{0})"""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            tools.extract_into_tensor(self.posterior_mean_coef_x0, t, x_start.shape)
            * x_start
            + tools.extract_into_tensor(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = tools.extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )

        assert (
            posterior_mean.shape[0] == posterior_variance.shape[0] == x_start.shape[0]
        )

        return posterior_mean, posterior_variance

    def p_mean_variance(
        self, model, x, t, clip_noise=True, denoised_fn=None, model_kwargs=None
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)

        if self.rescale_time_step:
            t = tools.scale_time_step(t, self.num_time_step)

        model_output = model(x, t, **model_kwargs)
