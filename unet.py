import os
import sys
import warnings
from abc import abstractmethod
from einops import rearrange, reduce, repeat

unet_path = os.getcwd()
sys.path.append(unet_path)

import torch.nn as nn
import torch.nn.functional as F
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
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=1
            )

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
            h = out_norm(h) * (1 + scale) + shift
            h = out_remain(h)
        else:
            h = h + embed_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class LinearAttentionBlock(nn.Module):
    """the implementation of attention according to official IDDPM,
        which is a little differene with ViT attention(timm package). For instance,
        ViT has dropout after proj_out and use LN on q and k after qkv project

    Args:
        nn (_type_): _description_
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, self.channels)
        self.qkv = nn.Conv1d(
            self.channels, self.channels * 3, kernel_size=1
        )  # equal to linear model: wx+b, it's ok to use nn.Linear
        self.proj_out = tools.zero_module(
            nn.Conv1d(self.channels, self.channels, kernel_size=1)
        )

    def forward(self, x):
        """

        Args:
            x (tensor): input of attentionblock

        Returns:
            tensor: output of attentionblock
        """
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


class DownSample(nn.Module):
    """downsample module of  unet"""

    def __init__(self, channels, use_conv=False):
        """
        Args:
            channels (int): input channels
            use_conv (bool): if true, use conv to downsample, else use avgpool
        """
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.down = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, channels, use_conv=False):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


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
        num_heads=1,
        use_cond_scale_shift=False,
        sample_use_conv=True,
    ):
        """_summary_

        Args:
            in_channels (int): the channels of input image, default=3
            out_channels (int): the channels of output iamge, default=3
            model_channels (int): base channel num of unet model, other channels is usually a multiple of model_channels
            num_classes (int): used to map class index to class_embed
            num_res_blocks (int): resblock number in each depth of unet
            attention_resolution (list): which resolution is need to attetion operation
            dropout (int, optional): Defaults to 0.
            channel_mult (tuple, optional): used to scale the model_channels to present the channel num in each resolution. Defaults to (1, 2, 4, 8).
            num_heads (int, optional):  the number of heads in attention. Defaults to 1.
            use_cond_scale_shift (bool, optional): introducing mechanisms of conditions, whether use scale and shift / use just add". Defaults to False.
            sample_use_conv (bool, optional): whether use learnable conv after F.interpolate. Defaults to True.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.num_res_blocks = num_res_blocks
        self.attention_resolution = attention_resolution
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_cond_scale_shift = use_cond_scale_shift
        self.sample_use_conv = sample_use_conv

        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
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
        input_block_channels = [current_channels]
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
                    layers.append(
                        LinearAttentionBlock(current_channels, self.num_heads)
                    )
                self.input_blocks.append(TimeOrRegularWapper(*layers))
                input_block_channels.append(current_channels)
            if depth != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimeOrRegularWapper(
                        DownSample(current_channels, self.sample_use_conv)
                    )
                )
                input_block_channels.append(current_channels)
                current_resolution *= 2

        self.middle_block = TimeOrRegularWapper(
            ResBlock(
                current_channels,
                time_embed_dim,
                self.dropout,
                use_cond_scale_shift=self.use_cond_scale_shift,
            ),
            LinearAttentionBlock(current_channels, num_heads=self.num_heads),
            ResBlock(
                current_channels,
                time_embed_dim,
                self.dropout,
                use_cond_scale_shift=self.use_cond_scale_shift,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        # print("input_block_channels")
        # print(input_block_channels)
        # :[128, 128, 128, 128, 256, 256, 256, 512, 512, 512, 1024, 1024]
        for depth, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                layers = [
                    ResBlock(
                        current_channels + input_block_channels.pop(),
                        time_embed_dim,
                        self.dropout,
                        out_channels=self.model_channels * mult,
                        use_cond_scale_shift=self.use_cond_scale_shift,
                    )
                ]
                current_channels = self.model_channels * mult
                if current_resolution in attention_resolution:
                    layers.append(
                        LinearAttentionBlock(current_channels, self.num_heads)
                    )
                if depth and i == self.num_res_blocks:  # when depth==0, don't upsample
                    layers.append(UpSample(current_channels, self.sample_use_conv))
                    current_resolution //= 2
                self.output_blocks.append(TimeOrRegularWapper(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            tools.zero_module(
                nn.Conv2d(model_channels, self.out_channels, kernel_size=3, padding=1)
            ),
        )

    def forward(self, x, timestep, y=None):
        assert (y is not None) == (self.num_classes is not None)
        down_history = []
        time_embed = self.time_embed(
            tools.timestep_embedding(timestep, self.model_channels)
        )

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            embed_all = time_embed + self.label_embed(y)
        else:
            embed_all = time_embed
        h = x
        for moudle in self.input_blocks:
            h = moudle(h, embed_all)
            down_history.append(h)

        h = self.middle_block(h, embed_all)

        for module in self.output_blocks:
            cat_in = th.cat([h, down_history.pop()], dim=1)
            h = module(cat_in, embed_all)
            print("-------")
            print(h[0, 0, 0:5, 0:5])

        return self.out(h)
