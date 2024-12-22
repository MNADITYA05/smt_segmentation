# segmentation/mmseg/builder.py
import warnings
from mmcv.utils import Registry

# Create a unique scope name to avoid conflicts
BACKBONES = Registry('backbone', scope='smt_seg')
NECKS = Registry('neck', scope='smt_seg')
HEADS = Registry('head', scope='smt_seg')
LOSSES = Registry('loss', scope='smt_seg')
SEGMENTORS = Registry('segmentor', scope='smt_seg')

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg are deprecated, '
            'please specify them in model config',
            UserWarning)
    return SEGMENTORS.build(cfg)