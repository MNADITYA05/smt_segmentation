import os
import torch
from collections import OrderedDict


def smt_convert(ckpt):
    """Convert SMT classification checkpoint to segmentation backbone format."""
    try:
        if not isinstance(ckpt, dict):
            raise TypeError(f'checkpoint must be a dict, got {type(ckpt)}')

        new_ckpt = OrderedDict()

        # Define prefix mappings for backbone
        prefix_mapping = {
            'patch_embed': 'backbone.patch_embed',
            'block': 'backbone.block',
            'norm': 'backbone.norm'
        }

        # Special handling for attention types
        for k, v in ckpt.items():
            # Skip classification head
            if k.startswith(('head.', 'fc.')):
                continue

            # Map keys to backbone
            mapped = False
            for old_prefix, new_prefix in prefix_mapping.items():
                if k.startswith(old_prefix):
                    new_k = k.replace(old_prefix, new_prefix, 1)
                    # Special handling for attention blocks
                    if 'attn' in new_k:
                        # Preserve both CA and SA attention weights
                        if any(x in new_k for x in ['q', 'k', 'v', 'kv']):
                            new_k = new_k  # Keep SA attention keys as is
                        elif any(x in new_k for x in ['s', 'local_conv']):
                            new_k = new_k  # Keep CA attention keys as is
                    new_ckpt[new_k] = v
                    mapped = True
                    break

            # Handle unmapped keys
            if not mapped:
                new_k = f'backbone.{k}'
                new_ckpt[new_k] = v

        return new_ckpt

    except Exception as e:
        raise RuntimeError(f'Error converting checkpoint: {str(e)}')


def load_checkpoint(filename, map_location=None):
    """Load checkpoint with enhanced error handling."""
    try:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'Checkpoint file {filename} does not exist')

        checkpoint = torch.load(filename, map_location=map_location)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            raise TypeError(f'checkpoint must be a dict, got {type(checkpoint)}')

        return state_dict

    except Exception as e:
        raise RuntimeError(f'Error loading checkpoint from {filename}: {str(e)}')


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint with atomic writing."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        checkpoint = {
            'meta': meta or {},
            'state_dict': model.state_dict()
        }

        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()

        tmp_filename = filename + '.tmp'
        torch.save(checkpoint, tmp_filename)
        os.replace(tmp_filename, filename)

    except Exception as e:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        raise RuntimeError(f'Error saving checkpoint: {str(e)}')


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state dict with detailed logging."""
    try:
        if logger is not None:
            logger.info(f'Loading state dict... Strict mode: {strict}')

        msg = module.load_state_dict(state_dict, strict=strict)

        if logger is not None:
            if msg.missing_keys:
                logger.warning(f'Missing keys: {msg.missing_keys}')
            if msg.unexpected_keys:
                logger.warning(f'Unexpected keys: {msg.unexpected_keys}')

        return msg

    except Exception as e:
        raise RuntimeError(f'Error loading state dict: {str(e)}')