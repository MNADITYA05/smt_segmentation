import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class SafeAdaptiveAvgPool2d(nn.Module):
    """Cross-platform adaptive average pooling with device handling."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x):
        try:
            if x.device.type == 'mps':
                # Handle MPS device
                x_cpu = x.cpu()
                pooled = F.adaptive_avg_pool2d(x_cpu, self.output_size)
                return pooled.to(x.device)
            return F.adaptive_avg_pool2d(x, self.output_size)
        except Exception as e:
            raise RuntimeError(f"Error in adaptive pooling: {str(e)}")


class UPerHead(BaseModule):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet`.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 pool_scales=(1, 2, 3, 6),
                 dropout_ratio=0.1,
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        # Input validation
        if not isinstance(in_channels, list):
            raise TypeError("in_channels must be a list")
        if not all(isinstance(c, int) and c > 0 for c in in_channels):
            raise ValueError("All channel numbers must be positive integers")
        if channels <= 0 or num_classes <= 0:
            raise ValueError("channels and num_classes must be positive")
        if not 0 <= dropout_ratio < 1:
            raise ValueError("dropout_ratio must be in [0, 1)")

        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.num_groups = 32  # Number of groups for GroupNorm

        # PSP Module
        self.psp_modules = nn.ModuleList()
        self.psp_convs = nn.ModuleList()

        for scale in pool_scales:
            self.psp_modules.append(SafeAdaptiveAvgPool2d(scale))
            self.psp_convs.append(
                ConvModule(
                    in_channels[-1],
                    channels,
                    1,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=self.num_groups, requires_grad=True),
                    act_cfg=dict(type='ReLU')
                )
            )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channel in self.in_channels[:-1]:  # Skip the last one
            l_conv = ConvModule(
                in_channel,
                channels,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=self.num_groups, requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=False
            )
            fpn_conv = ConvModule(
                channels,
                channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='GN', num_groups=self.num_groups, requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=False
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # FPN Bottleneck
        self.fpn_bottleneck = ConvModule(
            (len(pool_scales) + len(self.in_channels) - 1) * channels,
            channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=self.num_groups, requires_grad=True),
            act_cfg=dict(type='ReLU'),
            inplace=False
        )

        # Final classifier
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

    def psp_forward(self, inputs):
        """Forward function for PSP module."""
        x = inputs[-1]  # Take the deepest feature map
        psp_outs = []

        for pool_module, conv_module in zip(self.psp_modules, self.psp_convs):
            # Pool and convolve
            pooled = pool_module(x)
            conv = conv_module(pooled)

            # Upsample to match input size
            upsampled = F.interpolate(
                conv,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            psp_outs.append(upsampled)

        return torch.cat(psp_outs, dim=1)

    def _resize(self, x, target_size):
        """Safely resize feature maps."""
        if x.shape[2:] == target_size:
            return x
        return F.interpolate(
            x,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners
        )

    def forward(self, inputs):
        """Forward function."""
        # Input validation
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a list or tuple")
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} inputs, got {len(inputs)}")

        # Validate input shapes
        for i, (inp, channel) in enumerate(zip(inputs, self.in_channels)):
            if inp.size(1) != channel:
                raise ValueError(f"Input {i} has {inp.size(1)} channels, expected {channel}")

        try:
            # PSP Module forward
            psp_out = self.psp_forward(inputs)

            # FPN Module forward
            lateral_outputs = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                lateral_out = lateral_conv(inputs[i])
                lateral_outputs.append(lateral_out)

            # Feature fusion from top to bottom
            for i in range(len(lateral_outputs) - 1, -1, -1):
                if i > 0:
                    prev_shape = lateral_outputs[i - 1].shape[2:]
                    lateral_outputs[i - 1] = lateral_outputs[i - 1] + \
                                             self._resize(lateral_outputs[i], prev_shape)

            # FPN convolutions
            fpn_outputs = []
            for i in range(len(lateral_outputs)):
                fpn_out = self.fpn_convs[i](lateral_outputs[i])
                fpn_outputs.append(fpn_out)

            # Add PSP features
            fpn_outputs.append(psp_out)

            # Unify sizes to the first scale
            target_size = fpn_outputs[0].shape[2:]
            for i in range(1, len(fpn_outputs)):
                fpn_outputs[i] = self._resize(fpn_outputs[i], target_size)

            # Feature fusion and final prediction
            fused_features = torch.cat(fpn_outputs, dim=1)
            output = self.fpn_bottleneck(fused_features)
            output = self.conv_seg(output)

            return output

        except Exception as e:
            raise RuntimeError(f"Error in UPerNet forward pass: {str(e)}")

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)