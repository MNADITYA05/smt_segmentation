# utils/__init__.py
from .trainer import Trainer
from .metrics import AverageMeter, calculate_miou
from .checkpoint import load_checkpoint, save_checkpoint
from .dist_utils import setup_distributed, is_main_process
from .gradient_checkpointing import enable_gradient_checkpointing

# Import from root directory instead of relative imports
from logger import create_logger
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from .grad_cam import SegGradCAM
from .visualization import TrainingVisualizer

__all__ = [
    # Training components
    'Trainer',

    # Metrics
    'AverageMeter',
    'calculate_miou',

    # Logging
    'create_logger',

    # Checkpoint handling
    'load_checkpoint',
    'save_checkpoint',

    # Device and distributed setup
    'setup_distributed',
    'is_main_process',

    # Optimization
    'build_optimizer',
    'build_scheduler',

    # Memory management
    'enable_gradient_checkpointing',
    'SegGradCAM',

    'TrainingVisualizer'
]


