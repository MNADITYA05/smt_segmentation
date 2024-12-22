import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math
from utils.gradient_checkpointing import enable_gradient_checkpointing


class DWConv(nn.Module):
    """Depth-wise Convolution module with proper initialization and shape validation."""

    def __init__(self, dim=768):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")

        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(f"Input tensor shape mismatch: got {N}, expected {H * W}")

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP module with depth-wise convolution and residual connection."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if in_features <= 0 or hidden_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive")

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        """Forward pass with shape validation."""
        B, N, C = x.shape
        if C != self.in_features:
            raise ValueError(f"Input feature dimension mismatch: got {C}, expected {self.in_features}")

        identity = x
        x = self.fc1(x)

        # Add dwconv with residual connection
        conv_out = self.dwconv(x, H, W)
        if conv_out.shape != x.shape:
            raise ValueError(f"Shape mismatch after DWConv: {conv_out.shape} vs {x.shape}")

        x = self.act(x + conv_out)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # Validate output shape
        if x.shape != (B, N, self.out_features):
            raise ValueError(f"Output shape mismatch: got {x.shape}, expected {(B, N, self.out_features)}")

        return x


class Attention(nn.Module):
    """Multi-head attention module with channel and spatial attention mechanisms."""

    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        # Validate input parameters
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if ca_num_heads > 0 and dim % ca_num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by ca_num_heads {ca_num_heads}")
        if sa_num_heads > 0 and dim % sa_num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by sa_num_heads {sa_num_heads}")

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.split_groups = self.dim // ca_num_heads if ca_num_heads > 0 else self.dim

        # Initialize attention mechanisms
        if ca_attention == 1:
            self._init_channel_attention(dim, qkv_bias, expand_ratio)
        else:
            self._init_spatial_attention(dim, qkv_bias, qk_scale, attn_drop)

        self.apply(self._init_weights)

    def _init_channel_attention(self, dim, qkv_bias, expand_ratio):
        """Initialize channel attention components."""
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)

        # Initialize local convolutions
        for i in range(self.ca_num_heads):
            kernel_size = 3 + i * 2
            padding = 1 + i
            local_conv = nn.Conv2d(
                dim // self.ca_num_heads,
                dim // self.ca_num_heads,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                groups=dim // self.ca_num_heads
            )
            setattr(self, f"local_conv_{i + 1}", local_conv)

        # Projection layers
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, 1, 1, 0, groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, 1, 1, 0)

    def _init_spatial_attention(self, dim, qkv_bias, qk_scale, attn_drop):
        """Initialize spatial attention components."""
        head_dim = dim // self.sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def _init_weights(self, m):
        """Initialize layer weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__()

        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")

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
            expand_ratio=expand_ratio
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        """Forward pass with residual connections and shape validation."""
        B, N, C = x.shape

        # Store identity for residual
        identity = x

        # Attention branch
        attn_out = self.attn(self.norm1(x), H, W)
        if attn_out.shape != x.shape:
            raise ValueError(f"Attention output shape mismatch: {attn_out.shape} vs {x.shape}")
        x = identity + self.drop_path(attn_out)

        # MLP branch
        mlp_out = self.mlp(self.norm2(x), H, W)
        if mlp_out.shape != x.shape:
            raise ValueError(f"MLP output shape mismatch: {mlp_out.shape} vs {x.shape}")
        x = x + self.drop_path(mlp_out)

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlap."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        if any(x <= 0 for x in [img_size, patch_size, stride, in_chans, embed_dim]):
            raise ValueError("All dimensions must be positive")

        patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass with shape validation."""
        B, C, H, W = x.shape
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            raise ValueError(f"Input image dimensions ({H}, {W}) must be divisible by patch size {self.patch_size}")

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class SMT(nn.Module):
    """Scale-Aware Modulation Meet Transformer."""

    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[4, 4, 4, 2],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[3, 4, 18, 2],
                 ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2):
        super().__init__()

        # Input validation
        if len(embed_dims) != num_stages:
            raise ValueError(f"Length of embed_dims ({len(embed_dims)}) must match num_stages ({num_stages})")

        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        self.gradient_checkpointing = False

        # Calculate drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Build stages
        for i in range(num_stages):
            # Create patch embedding layer
            if i == 0:
                patch_embed = self._create_head(head_conv, embed_dims[i])
            else:
                patch_embed = self._create_patch_embed(i, img_size, in_chans, embed_dims)

            # Create transformer blocks
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
                    expand_ratio=expand_ratio
                )
                for j in range(depths[i])
            ])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            # Register components
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _create_head(self, head_conv, dim):
        """Create the initial head of the network."""
        return nn.Sequential(
            nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv == 7 else 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        )

    def _create_patch_embed(self, stage_idx, img_size, in_chans, embed_dims):
        """Create patch embedding layer for intermediate stages."""
        return OverlapPatchEmbed(
            img_size=img_size if stage_idx == 0 else img_size // (2 ** (stage_idx + 1)),
            patch_size=7 if stage_idx == 0 else 3,
            stride=4 if stage_idx == 0 else 2,
            in_chans=in_chans if stage_idx == 0 else embed_dims[stage_idx - 1],
            embed_dim=embed_dims[stage_idx]
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Extract hierarchical features with shape validation."""
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)

            # Apply transformer blocks with optional gradient checkpointing
            for blk in block:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, H, W)
                else:
                    x = blk(x, H, W)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return tuple(outs)

    def forward(self, x):
        """Forward pass returning multi-scale features."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")

        return self.forward_features(x)

    def train(self, mode=True):
        """Override train mode to handle gradient checkpointing."""
        super().train(mode)
        if mode and self.gradient_checkpointing:
            # Enable gradient checkpointing for memory efficiency
            for block in self.modules():
                if isinstance(block, Block):
                    block.train(True)
        return self