import argparse
import DM.utils as tools
import DM.gaussian_diffusion as gd

from DM import dist_utils, logger
from DM.train import TrainLoop
from DM.unet import Unet

NUM_CLASSES = 1000


def main():
    args = create_argparser().parse_args()

    # init distributed environment
    dist_utils.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        save_interval=10000,
        resume_checkpoint="",
    )
    # add model and diffusion config in defaults
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    tools.add_dict_to_argparser(parser, defaults)
    return parser


def model_and_diffusion_defaults():
    return dict(
        image_size=32,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolution="16,8",
        dropout=0.0,
        learn_sigma=False,
        simga_small=False,
        class_cond=False,
        diffusion_step=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """

    Args:
        image_size (int): 256/64/32
        num_channels (int): model channel of unet
        num_res_blocks (int): number of resbloks in same resolution
        learn_sigma (bool): whether use learnable var/std
        class_cond (bool): whether has class condition
        attention_resolutions (string): like "16,8"                                                                                                                                  ): _description_
        num_heads (int): attention head
        num_heads_upsample (int): _description_
        use_scale_shift_norm (bool):  introducing mechanisms of conditions, whether use scale and shift / use just add". Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    return Unet(
        in_channels=3,
        out_channels=(3 if not learn_sigma else 6),
        model_channels=num_channels,
        num_classes=(NUM_CLASSES if class_cond else None),
        num_res_blocks=num_res_blocks,
        attention_resolution=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads,
        use_cond_scale_shift=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Args:
        steps (int, optional): time steps. Defaults to 1000.
        learn_sigma (bool, optional): whether use learnable var/std. Defaults to False.
        sigma_small (bool, optional): if not learnable var/std, whether small or large const var/std. Defaults to False.
        noise_schedule (str, optional): Defaults to "linear".
        use_kl (bool, optional): whether use kl or mse as loss. Defaults to False.
        predict_xstart (bool, optional): _description_. Defaults to False.
        rescale_timesteps (bool, optional): _description_. Defaults to False.
        rescale_learned_sigmas (bool, optional): if True, use mse + rescale kl. Defaults to False.
        timestep_respacing (str, optional): _description_. Defaults to "".
    """
    betas = tools.create_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
