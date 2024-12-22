import warnings
import torch
import torch.nn as nn
from functools import partial
from mmcv.runner import BaseModule, _load_checkpoint, load_state_dict
from mmcv.utils import to_2tuple
from logger import create_logger
from ..builder import BACKBONES
from ..utils.ckpt_convert import smt_convert


class DWConv(nn.Module):
    """Depth-wise Convolution with MMCV compatibility."""

    def __init__(self, dim=768):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")

        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(f"Input tensor shape mismatch: N={N}, H*W={H * W}")

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(BaseModule):
    """MLP module with MMCV BaseModule inheritance."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 init_cfg=None):
        super().__init__(init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(BaseModule):
    """Attention module with channel and spatial attention mechanisms."""

    def __init__(self,
                 dim,
                 ca_num_heads=4,
                 sa_num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__(init_cfg)

        # Validate input dimensions
        if dim <= 0 or (ca_num_heads > 0 and dim % ca_num_heads != 0) or \
                (sa_num_heads > 0 and dim % sa_num_heads != 0):
            raise ValueError("Invalid dimension or number of heads")

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads
        self.split_groups = self.dim // ca_num_heads if ca_num_heads > 0 else self.dim

        # Common components
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if ca_attention == 1:
            self._init_channel_attention(dim, qkv_bias, expand_ratio)
        else:
            self._init_spatial_attention(dim, qkv_bias, qk_scale, attn_drop)

    def _init_channel_attention(self, dim, qkv_bias, expand_ratio):
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)

        for i in range(self.ca_num_heads):
            conv = nn.Conv2d(
                dim // self.ca_num_heads,
                dim // self.ca_num_heads,
                kernel_size=(3 + i * 2),
                padding=(1 + i),
                stride=1,
                groups=dim // self.ca_num_heads
            )
            setattr(self, f"local_conv_{i + 1}", conv)

        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, 1, 1, 0, groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, 1, 1, 0)

    def _init_spatial_attention(self, dim, qkv_bias, qk_scale, attn_drop):
        head_dim = dim // self.sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        if self.ca_attention == 1:
            return self._channel_attention_forward(x, H, W)
        else:
            return self._spatial_attention_forward(x, H, W)

    def _channel_attention_forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)

        s_out = []
        for i in range(self.ca_num_heads):
            conv = getattr(self, f"local_conv_{i + 1}")
            s_i = conv(s[i]).reshape(B, self.split_groups, -1, H, W)
            s_out.append(s_i)

        s_out = torch.cat(s_out, 2).reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out)))).reshape(B, C, N).permute(0, 2, 1)

        x = s_out * v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _spatial_attention_forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.dw_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                       N).transpose(1,
                                                                                                                    2)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """Transformer block with MMCV compatibility."""

    def __init__(self,
                 dim,
                 ca_num_heads,
                 sa_num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            ca_num_heads=ca_num_heads,
            sa_num_heads=sa_num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            ca_attention=ca_attention,
            expand_ratio=expand_ratio,
            init_cfg=None
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            init_cfg=None
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding with overlap and MMCV compatibility."""

    def __init__(self,
                 img_size=224,
                 patch_size=3,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Head(BaseModule):
    """Initial processing head with MMCV compatibility."""

    def __init__(self, dim, head_conv, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, head_conv, 2,
                      padding=1 if head_conv == 3 else 3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, dim, kernel_size=2, stride=2)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


@BACKBONES.register_module()
class SMT(BaseModule):
    """MMCV-compatible SMT implementation for segmentation."""

    def __init__(self,
                 pretrain_img_size=224,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1],
                 sa_num_heads=[-1, -1, 8, 16],
                 mlp_ratios=[4, 4, 4, 2],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3],
                 ca_attentions=[1, 1, 1, 0],
                 num_stages=4,
                 head_conv=3,
                 expand_ratio=2,
                 pretrained=None,
                 init_cfg=None,
                 convert_weights=True):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.convert_weights = convert_weights
        self.depths = depths
        self.num_stages = num_stages
        self.pretrain_img_size = pretrain_img_size
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]

        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = False

        # Build stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(
                    dim=embed_dims[0],
                    head_conv=head_conv,
                    init_cfg=None
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    init_cfg=None
                )

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    ca_num_heads=ca_num_heads[i],
                    sa_num_heads=sa_num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    ca_attention=0 if i == 2 and j % 2 != 0 else ca_attentions[i],
                    expand_ratio=expand_ratio,
                    init_cfg=None
                )
                for j in range(depths[i])
            ])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        """Initialize weights with proper logging."""
        logger = create_logger(output_dir='work_dirs', name='smt')

        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, '
                           f'training start from scratch')
            return

        if not isinstance(self.init_cfg, dict) or 'checkpoint' not in self.init_cfg:
            logger.warning(f'Only support specify `Pretrained` in `init_cfg` in '
                           f'{self.__class__.__name__}')
            return

        try:
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'],
                logger=logger,
                map_location='cpu'
            )

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if self.convert_weights:
                state_dict = smt_convert(state_dict)

            load_state_dict(self, state_dict, strict=False, logger=logger)
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            raise

    def forward(self, x):
        """Forward function with optional gradient checkpointing."""
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)

            # Apply blocks with gradient checkpointing if enabled
            for blk in block:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, H, W)
                else:
                    x = blk(x, H, W)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        """Override train mode to handle gradient checkpointing."""
        super().train(mode)
        if mode and self.gradient_checkpointing:
            def enable_checkpointing(m):
                if hasattr(m, 'gradient_checkpointing'):
                    m.gradient_checkpointing = True

            self.apply(enable_checkpointing)
        return self