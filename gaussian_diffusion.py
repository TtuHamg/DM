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
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_time_step=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_time_step = rescale_time_step

        self.betas = np.array(betas, dtype=np.float64)
        assert len(self.betas.shape) == 1
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        self.num_time_step = int(betas.shape[0])
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1 - self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_invm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # q_posterior: q(x_{t-1} | x_{t}, x_{0})
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
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
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
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """get p(x_{t-1} | x_{t}) mean and var, as well as a prediction of the initial x_0

        Args:
            model (nn.Module): default unet
            x (tensor): x_t
            t (_type_): time step
        Raises:
            NotImplementedError: model mean type not implementation

        Returns:
            dict: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        if self.rescale_time_step:
            t = tools.scale_time_step(t, self.num_time_step)

        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]:  # Learned variance(IDDPM)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                # in this case, model_var_values indicate learned log variance
                model_log_variance = model_var_values
                model_variance = th.exp(model_var_values)
            else:
                # in this case, model_var_values indicate learnable coef V
                min_log = tools.extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = tools.extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                coef_V = (model_var_values + 1) / 2
                model_log_variance = coef_V * max_log + (1 - coef_V) * min_log
                model_variance = th.exp(model_log_variance)
        else:  # Fix variance(DDPM)
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                # FIXME:why use posterior_variance in official IDDPM
                # model_variance = np.append(self.posterior_variance[1], self.betas[1:])
                model_variance = tools.extract_into_tensor(self.beta, t, x.shape)
                model_log_variance = tools.extract_into_tensor(
                    np.log(self.betas), t, x.shape
                )
            else:
                model_variance = tools.extract_into_tensor(
                    self.posterior_variance, t, x.shape
                )
                model_log_variance = tools.extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )

        def process_x_start(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xtart = process_x_start(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xtart = process_x_start(model_output)
            else:
                pred_xtart = process_x_start(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _ = self.q_posterior_mean_variance(
                x_start=pred_xtart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape
            == model_log_variance.shape
            == model_variance.shape
            == pred_xtart.shape
            == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xtart,
        }

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """x_{t-1}=prosterior_mean_coef_xt * x_{t} +prosterior_mean_coef_x0 * x_{0}

        Args:
            x_t (tensor): x_{t}
            xprev (tensor): x_{t-1}: the output of unet

        Returns:
            tensor: x_{0}
        """
        assert x_t.shape == xprev.shape
        return tools.extract_into_tensor(
            1.0 / self.posterior_mean_coef_x0, t, x_t.shape
        ) - tools.extract_into_tensor(
            self.posterior_mean_coef_xt / self.posterior_mean_coef_x0, t, x_t.shape
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """x_{t}=sqrt_alphas_cumprod * x_{0} + sqrt_beta_cumprod * eps

        Args:
            x_t (tensor): x_{t}
            eps (tensor): epsilon

        Returns:
            tensor: x_{0}
        """
        assert x_t.shape == eps.shape
        return (
            tools.extract_into_tensor(self.sqrt_inv_alphas_cumprod, t, x_t.shape) * x_t
            - tools.extract_into_tensor(self.sqrt_invm1_alphas_cumprod, t, x_t.shape)
            * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """x_{t}=sqrt_alphas_cumprod * x_{0} + sqrt_beta_cumprod * eps

        Args:
            x_t (tensor): x_{t}
            pred_xstart (tensor): x_start predicted by model

        Returns:
            tensor: epsilon
        """
        assert x_t.shape == pred_xstart.shape
        return (
            tools.extract_into_tensor(self.sqrt_inv_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / tools.extract_into_tensor(self.sqrt_invm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """sample x_{t-1} from x_{t}, which t is given in the params

        Args:
            model (nn.Module): default unet
            x (tesor): x_{t}
            t (tuple/list): time step
            clip_denoised (True, optional): whether clip x_start to [-1, 1]. Defaults to True.
            denoised_fn (False, optional):  a function which applies to the x_start prediction before it is used to sample. Defaults to None.
            model_kwargs (_type_, optional): _description_. Defaults to None.

        Returns:
            a dict containing the following keys:
            - 'sample': a random sample x_{t-1} from the model.
            - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().unsqueeze(-1)  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """generate samples from the model

        Args:
            model (nn.Module): default unet
            shape (_type_): the shape of the samples, (N, C, H, W)
            noise (tensor, optional): x_{T}. Defaults to None.
            device (_type_, optional): if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.. Defaults to None.
            progress (bool, optional): whether use tqdm to show process. Defaults to False.

        Returns:
            _type_: _description_
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise,
            clip_denoised,
            denoised_fn,
            model_kwargs,
            device,
            progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device == None:
            device = next(model.parameters()).device  # parameters() returns a iterator
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise  # x_{T}
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_time_step))[::-1]

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)  # each img in batch has t
            with th.no_grad():
                out = self.p_sample(
                    model, img, t, clip_denoised, denoised_fn, model_kwargs
                )
                yield out
                img = out["sample"]  # x_{t-1} is regarded as next loop's x_{t}

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """DDIM sample. When sigma is implemented as follows, it degraded to DDPM.

        Args:
            model (nn.Module): default unet
            x (tensor): x_{t}

        Returns:
            tensor: x_{t-1}
        """
        out = self.p_mean_variance(
            model, x, t, clip_denoised, denoised_fn, model_kwargs
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = tools.extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = tools.extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = th.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) * th.sqrt(
            1 - alpha_bar / alpha_bar_prev
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )

        nonzero_mask = (t != 0).float().unsqueeze(-1)  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return {"sample": sample, "pred_xstart:": out["pred_xstart"]}

    def ddim_deterministic_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """different from official IDDPM. When sigma=0, it's a deterministic sample

        Args:
            model (nn.Module): default unet
            x (tensor): x_{t}

        Returns:
            tensor: x_{t-1}
        """
        out = self.p_mean_variance(
            model, x, t, clip_denoised, denoised_fn, model_kwargs
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = tools.extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = tools.extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        final = None
        pass

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device == None:
            device = next(model.parameters()).device  # parameters() returns a iterator
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise  # x_{T}
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_time_step))[::-1]

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)  # each img in batch has t
            with th.no_grad():
                out = self.ddim_sample(
                    model, img, t, clip_denoised, denoised_fn, model_kwargs
                )
                yield out
                img = out["sample"]  # x_{t-1} is regarded as next loop's x_{t}
