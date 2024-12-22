import torch
import torch.nn as nn
from segmentation.mmseg.backbones.smt import SMT
from segmentation.mmseg.decode_heads.upernet import UPerHead
from segmentation.mmseg.utils.ckpt_convert import smt_convert


def remap_pretrained_keys(state_dict):
    """
    Remap pretrained keys to match the current model structure.
    Handles both backbone and decode_head keys properly.
    """
    try:
        new_state_dict = {}

        # Define key mappings for different parts of the model
        backbone_keys = {
            'patch_embed': 'backbone.patch_embed',
            'block': 'backbone.block',
            'norm': 'backbone.norm',
        }

        for k, v in state_dict.items():
            # Skip classification head weights
            if k.startswith(('head.', 'fc.')):
                continue

            # Handle backbone keys
            new_k = k
            for old_prefix, new_prefix in backbone_keys.items():
                if k.startswith(old_prefix):
                    new_k = k.replace(old_prefix, new_prefix, 1)
                    break

            if not any(k.startswith(prefix) for prefix in ['backbone.', 'decode_head.']):
                new_k = f'backbone.{k}'

            new_state_dict[new_k] = v

        return new_state_dict
    except Exception as e:
        raise RuntimeError(f"Error remapping pretrained keys: {str(e)}")


import torch
import torch.nn as nn
from segmentation.mmseg.backbones.smt import SMT
from segmentation.mmseg.decode_heads.upernet import UPerHead
from segmentation.mmseg.utils.ckpt_convert import smt_convert


def remap_pretrained_keys(state_dict):
    """Remap pretrained keys to match the current model structure."""
    try:
        new_state_dict = {}

        # Handle backbone keys
        for k, v in state_dict.items():
            # Skip classification head weights
            if k.startswith(('head.', 'fc.')):
                continue

            # Add backbone prefix if not present
            if not k.startswith('backbone.'):
                new_k = f'backbone.{k}'
            else:
                new_k = k

            new_state_dict[new_k] = v

        return new_state_dict

    except Exception as e:
        raise RuntimeError(f"Error remapping pretrained keys: {str(e)}")


def build_model(config, device, logger):
    """Build SMT model with UPerNet decoder."""
    try:
        # Create backbone
        logger.info("Building SMT backbone...")
        backbone = SMT(
            embed_dims=config.MODEL.SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.SMT.QKV_BIAS,
            depths=config.MODEL.SMT.DEPTHS,
            ca_attentions=config.MODEL.SMT.CA_ATTENTIONS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )

        # Load pretrained weights if specified
        if config.MODEL.PRETRAINED:
            logger.info(f"Loading pretrained weights from {config.MODEL.PRETRAINED}")
            try:
                checkpoint = torch.load(config.MODEL.PRETRAINED, map_location=device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Always convert classification weights to segmentation format
                state_dict = smt_convert(state_dict)
                state_dict = remap_pretrained_keys(state_dict)

                # Load state dict
                msg = backbone.load_state_dict(state_dict, strict=False)
                logger.info(f"Pretrained weight loading: {msg}")

                # Clear memory
                del checkpoint
                del state_dict
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Error loading pretrained weights: {str(e)}")
                logger.warning("Training will continue with random initialization")

        # Create UPerNet decoder
        logger.info("Building UPerNet decoder...")
        decoder = UPerHead(
            in_channels=config.MODEL.DECODE_HEAD.IN_CHANNELS,
            channels=config.MODEL.DECODE_HEAD.CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            dropout_ratio=config.MODEL.DECODE_HEAD.DROPOUT_RATIO,
            align_corners=config.MODEL.DECODE_HEAD.ALIGN_CORNERS
        )

        # Create segmentation model
        class SegmentationModel(nn.Module):
            def __init__(self, backbone, decoder):
                super().__init__()
                self.backbone = backbone
                self.decode_head = decoder
                self.gradient_checkpointing = False

            def forward(self, x):
                features = self.backbone(x)
                if not isinstance(features, (tuple, list)):
                    features = [features]
                out = self.decode_head(features)
                return out

        # Create and move model to device
        model = SegmentationModel(backbone, decoder)
        model = model.to(device)

        # Enable gradient checkpointing if configured
        if getattr(config.TRAIN, 'GRADIENT_CHECKPOINTING', False):
            model.gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled")

        return model

    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise