import os
import sys
import warnings
from abc import abstractmethod
from einops import rearrange, reduce, repeat

unet_path = os.getcwd()
sys.path.append(unet_path)

import torch.nn as nn
import torch as th
import utils as tools


class TimeResBlock(nn.Module):
    @abstractmethod
    def forward(self, x, embed):
        """any child module has to write forward method"""


class TimeOrRegularWapper(nn.Sequential, TimeResBlock):
    """ "Unified ResBlock with Embedding and Regular Convolution without Embedding"lock (_type_): _description_"""

    def forward(self, x, embed):
        for layer in self:
            if isinstance(layer, TimeResBlock):
                x = layer(x, embed)
            else:
                x = layer(x)
        return x


class ResBlock(TimeResBlock):
    """resblock in unet

    notes: groupnorm's param: affine default true. It's learnable params scale and shift belongs to norm
    which can't learn the time feature.
    """

    def __init__(
        self,
        in_channels,
        embed_channels,
        dropout,
        out_channels=None,
        use_cond_scale_shift=False,
        use_conv=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_cond_scale_shift = use_cond_scale_shift
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                embed_channels,
                2 * self.out_channels if use_cond_scale_shift else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            tools.zero_module(
                nn.Conv2d(
                    self.out_channels, self.out_channels, kernel_size=3, padding=1
                )
            ),
        )

        if self.out_channels == self.in_channels:
            if self.use_conv:
                warnings.warn(
                    f"channels in resblock is consistent, suggest not to use_conv.",
                    category=Warning,
                )
            self.skip_connection = nn.Identity()
        elif self.use_conv:
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=3, padding=1
            )
        else:
            raise ValueError(f"Either keep channels consistent or use_conv.")

    def forward(self, x, embed):
        """forward process about unet

        Args:
            x (tensor): unet input
            embed (tensor): time_embed, maybe contain label embed which processed before resblock

        Returns:
            tensor: output of resblock
        """
        h = self.in_layers(x)
        embed_out = self.embed_layers(embed).type(h.dtype)
        while len(embed_out.shape) < len(h.shape):
            embed_out = embed_out[..., None]
        if self.use_cond_scale_shift:
            out_norm, out_remain = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(embed_out, 2, dim=1)
            h = out_norm(h) * scale + shift
            h = out_remain(h)
        else:
            h = h + embed_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(
            channels, channels * 3, kernel_size=1
        )  # equal to linear model: wx+b
        self.proj_out = tools.zero_module(
            nn.Conv1d(self.channels, self.channels, kernel_size=1)
        )

    def forward(self, x):
        _, _, H, W = x.shape  # batch, channel, (H,W)
        x = rearrange(x, "b c x y->b c (x y)")
        qkv = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d -> b h c d", h=self.num_heads), qkv
        )
        _, _, head_dims, _ = q.shape
        scale = head_dims**-0.5
        atten = th.einsum("b h c t, b h c s->b h t s", q, k) * scale
        atten = th.softmax(atten, dim=-1)
        h = th.einsum("b h t s, b h c s-> b h c t", atten, v)
        h = rearrange(h, "b h c t->b (h c) t")
        h = self.proj_out(h)
        return rearrange(x + h, "b c (h w)->b c h w", h=H, w=W)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_classes,
        num_res_blocks,
        attention_resolution,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        use_cond_scale_shift=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.num_res_blocks = num_res_blocks
        self.attention_resolution = attention_resolution
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_cond_scale_shift = use_cond_scale_shift

        time_embed_dim = self.model_channels * 4
        self.tim_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.model_channels, time_embed_dim),
        )

        if num_classes is not None:
            self.label_embed = nn.Embedding(
                num_classes, time_embed_dim
            )  # DDPM paper use label_embed+time+embed

        self.input_blocks = nn.ModuleList(
            [
                TimeOrRegularWapper(
                    nn.Conv2d(
                        self.in_channels, self.model_channels, kernel_size=3, padding=1
                    )
                )
            ]
        )

        current_channels = model_channels
        current_resolution = 1
        for depth, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        current_channels,
                        time_embed_dim,
                        self.dropout,
                        out_channels=mult * self.model_channels,
                        use_cond_scale_shift=self.use_cond_scale_shift,
                    )
                ]
                current_channels = mult * self.model_channels
                if current_resolution in self.attention_resolution:
                    layers.append()
