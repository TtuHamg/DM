import torch as th
import torch.nn as nn
from itertools import repeat
from functools import reduce
import collections.abc
import math

import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp, PatchEmbed  # Attention

import torch.nn.functional as F
import random

import einops
import numpy as np

# try:
#     import xformers
#     import xformers.ops

#     ATTENTION = "xformer"
# except:
#     ATTENTION = "vanilla"


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return th.FloatTensor(sinusoid_table).unsqueeze(0)  # shape:(1, n_position, d_hid)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(
        imgs, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=patch_size, p2=patch_size
    )
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(
        x, "B (h w) (p1 p2 C)->B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size
    )
    return x


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        atten_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads if atten_head_dim is None else atten_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(th.zeros(all_head_dim))
            self.v_bias = nn.Parameter(th.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.atten_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = th.cat(
                (
                    self.q_bias,
                    th.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        # B, N 3C -> 3 B num_heads N C
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # B, num_heads, N, N
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.atten_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj_drop(self.proj(x))

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
        dropout=0.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_dropout,
            proj_drop=proj_dropout,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(th.cat([x, skip], dim=-1))
        x = x + self.drop_out(self.attn(self.norm1(x)))
        x = x + self.drop_out(self.mlp(self.norm2(x)))
        return x


class UViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        norm_layer=nn.LayerNorm,
        mlp_time_embed=False,
        num_classes=-1,
        dropout=0.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        if self.num_classes > 0:
            self.label_embed = nn.Embedding(self.num_classes, embed_dim)
            self.uncond_embed = nn.Parameter(th.zeros(1, embed_dim))
            trunc_normal_(self.uncond_embed, std=0.02)
            self.extras = 2  # time token and class token
        else:
            self.extras = 1  # time token

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.in_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                )
                for _ in range(depth // 2)
            ]
        )

        self.middle_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            dropout=dropout,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    skip=True,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                )
                for _ in range(depth // 2)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.encoder_pos_embed = nn.Parameter(
            th.zeros(1, self.extras, self.num_patches, embed_dim)
        )

        trunc_normal_(self.encoder_pos_embed, std=0.02)
        self.apply(self._init_weights)  # only initialize linear layer and layernorm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 0)

    def forward(self, x, timesteps, mask=None, y=None):
        x = self.patch_embed(x)

        # time condition, (N, C) -> (N, 1, C)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(1)
        x = th.cat([time_token, x], dim=1)

        # class condition
        if self.extras == 2:
            if y is None:
                # use unconditional embedding
                B, *_ = x.shape
                label_embed = (
                    self.uncond_embed.expand(B, -1, -1).type_as(x).to(x.device)
                )
            else:
                label_embed = self.label_embed(y)
                label_embed = label_embed.unsqueeze(dim=1)
            x = th.cat([label_embed, x], dim=1)

        x = x + self.encoder_pos_embed

        B, N, C = x.shape
        # mask will skip time_token, etc
        if mask is not None:
            x = x[~mask].reshape(B, -1, C)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.middle_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        return x
