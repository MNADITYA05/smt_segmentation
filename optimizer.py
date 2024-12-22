import torch
from torch import optim
from functools import partial
import logging

try:
    from apex.optimizers import FusedAdam, FusedLAMB

    HAS_APEX = True
except ImportError:
    FusedAdam = None
    FusedLAMB = None
    HAS_APEX = False
    logging.info("APEX not found. FusedAdam and FusedLAMB optimizers will not be available.")


def get_layer_id(name, num_backbone_layers=4):
    """Determine layer index for layer-wise learning rate decay.

    Args:
        name (str): Parameter name
        num_backbone_layers (int): Number of layers in backbone

    Returns:
        int: Layer index for learning rate scaling
    """
    if 'backbone' not in name:
        return num_backbone_layers  # Decoder gets the final layer rate

    if 'layer' in name:
        try:
            layer_id = int(name.split('.')[2])  # Format: backbone.layer1.xxx
            return min(layer_id, num_backbone_layers - 1)
        except (IndexError, ValueError):
            return 0
    return 0  # Default to first layer if structure unclear


def get_param_groups(model, config):
    """Create parameter groups with layer-wise decay and proper weight decay settings.

    Args:
        model: The model containing backbone and decoder
        config: Configuration object containing learning rates and other settings
    """
    # Parameters that should not use weight decay
    no_weight_decay_list = {
        'absolute_pos_embed',
        'relative_position_bias_table',
        'norm',
        'bias',
        'bn'
    }

    # Calculate layer-wise learning rate scales
    num_layers = config.MODEL.BACKBONE.NUM_LAYERS
    layer_scales = [
        config.TRAIN.BASE_LR * (config.TRAIN.LAYER_DECAY ** i)
        for i in range(num_layers + 1)
    ]

    param_groups = []
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Ensure each parameter is processed only once
        if param in seen_params:
            continue
        seen_params.add(param)

        # Determine if parameter should have weight decay
        no_decay = any(nd in name.lower() for nd in no_weight_decay_list)

        # Get layer index for learning rate scaling
        layer_id = get_layer_id(name, num_layers)

        # Determine if parameter is part of backbone
        is_backbone = 'backbone' in name

        # Create descriptive group name
        group_name = f"{'backbone' if is_backbone else 'decoder'}_" \
                     f"layer_{layer_id}_{'no_' if no_decay else ''}decay"

        # Calculate learning rate with layer-wise decay and backbone multiplier
        lr = layer_scales[layer_id]
        if is_backbone:
            lr *= config.TRAIN.BACKBONE_LR_MULTIPLIER

        param_groups.append({
            'name': group_name,
            'params': [param],
            'lr': lr,
            'weight_decay': 0.0 if no_decay else config.TRAIN.WEIGHT_DECAY,
            'amsgrad': config.TRAIN.OPTIMIZER.AMSGRAD
        })

    # Consolidate groups with the same settings
    consolidated_groups = {}
    for group in param_groups:
        key = (group['lr'], group['weight_decay'], group['amsgrad'])
        if key not in consolidated_groups:
            consolidated_groups[key] = {
                'params': [],
                'lr': group['lr'],
                'weight_decay': group['weight_decay'],
                'amsgrad': group['amsgrad'],
                'name': group['name']
            }
        consolidated_groups[key]['params'].extend(group['params'])

    return list(consolidated_groups.values())


def build_optimizer(config, param_groups):
    """Build optimizer with specified parameters.

    Args:
        config: Configuration object
        param_groups: Parameter groups with proper learning rates and decay settings
    """
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()

    optimizer_args = {
        'params': param_groups,
        'lr': config.TRAIN.BASE_LR,
        'weight_decay': config.TRAIN.WEIGHT_DECAY
    }

    if opt_lower in ['adamw', 'fusedadam', 'fusedlamb']:
        optimizer_args.update({
            'betas': config.TRAIN.OPTIMIZER.BETAS,
            'eps': config.TRAIN.OPTIMIZER.EPS
        })
    elif opt_lower == 'sgd':
        optimizer_args.update({
            'momentum': config.TRAIN.OPTIMIZER.MOMENTUM,
            'nesterov': True
        })

    if opt_lower == 'adamw':
        optimizer = optim.AdamW(**optimizer_args)
    elif opt_lower == 'fusedadam' and HAS_APEX:
        optimizer = FusedAdam(**optimizer_args)
    elif opt_lower == 'fusedlamb' and HAS_APEX:
        optimizer = FusedLAMB(**optimizer_args)
    elif opt_lower == 'sgd':
        optimizer = optim.SGD(**optimizer_args)
    else:
        available_optimizers = ['adamw', 'sgd']
        if HAS_APEX:
            available_optimizers.extend(['fusedadam', 'fusedlamb'])
        raise ValueError(
            f'Unknown optimizer: {opt_lower}. '
            f'Available optimizers: {", ".join(available_optimizers)}'
        )

    return optimizer


def log_optimizer_info(param_groups, logger=None):
    """Log optimizer parameter group information.

    Args:
        param_groups: Parameter groups to log
        logger: Optional logger instance
    """
    total_params = 0
    msg = ['Optimizer parameter groups:']

    for group in param_groups:
        params = sum(p.numel() for p in group['params'])
        total_params += params
        msg.append(
            f"- {group['name']}: {params:,} parameters\n"
            f"  lr: {group['lr']:.2e}, "
            f"weight_decay: {group['weight_decay']:.2e}, "
            f"amsgrad: {group['amsgrad']}"
        )

    msg.append(f"\nTotal parameters: {total_params:,}")
    msg = '\n'.join(msg)

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return total_params


def build_optimizer_with_model(config, model, logger=None):
    """Convenience function to build optimizer directly from model.

    Args:
        config: Configuration object
        model: Model to optimize
        logger: Optional logger for parameter information

    Returns:
        torch.optim.Optimizer: Configured optimizer instance
    """
    param_groups = get_param_groups(model, config)

    if logger:
        log_optimizer_info(param_groups, logger)

    return build_optimizer(config, param_groups)