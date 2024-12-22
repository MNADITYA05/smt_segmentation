from .smt import SMT, OverlapPatchEmbed, Block, Attention, Mlp, DWConv, Head

# Define all public interfaces that can be imported from this module
__all__ = [
    'SMT',  # Main SMT model class
    'OverlapPatchEmbed',  # Patch embedding module
    'Block',  # Transformer block
    'Attention',  # Attention mechanism
    'Mlp',  # MLP module
    'DWConv',  # Depthwise convolution
    'Head',  # Initial processing head
]

# Register backbone model mapping
backbone_entrypoints = {
    'smt': SMT,  # Entry point for SMT model
}


def get_backbone(backbone_name, **kwargs):
    """
    Get backbone model by name.
    Args:
        backbone_name (str): Name of the backbone model
        **kwargs: Additional arguments to pass to the model
    Returns:
        nn.Module: Backbone model
    """
    if backbone_name not in backbone_entrypoints:
        raise ValueError(f'Unknown backbone: {backbone_name}')

    return backbone_entrypoints[backbone_name](**kwargs)