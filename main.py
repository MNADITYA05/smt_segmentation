import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import psutil
from torch.utils.data import DataLoader
from tqdm import tqdm

# Config
from config import get_config

# Utils
from utils.dist_utils import (
    setup_distributed,
    is_main_process,
    NativeScalerWithGradNormCount
)
from logger import create_logger
from optimizer import build_optimizer_with_model

from utils.trainer import Trainer
from utils.checkpoint import load_checkpoint, save_checkpoint, auto_resume_helper
from utils.metrics import calculate_miou
from utils.gradient_checkpointing import enable_gradient_checkpointing
from utils.gradient_checkpointing import MemoryManager, GradientCheckpointManager

# Data and models
from models import build_model
from data.dataset import CustomSegmentationDataset
from optimizer import build_optimizer
from lr_scheduler import build_scheduler  # Add this import


def get_memory_info():
    """Get system memory information"""
    vm = psutil.virtual_memory()
    return {
        'total': vm.total / (1024 ** 3),  # GB
        'available': vm.available / (1024 ** 3),  # GB
        'percent_used': vm.percent
    }


def setup_device(config, logger):
    """Enhanced device setup with memory management"""
    device = None
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
            # Set MPS memory limit
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")

        if not device:
            raise RuntimeError("No suitable device found")

        # Memory management
        memory_info = get_memory_info()
        logger.info(f"System Memory - Total: {memory_info['total']:.2f}GB, "
                   f"Available: {memory_info['available']:.2f}GB, "
                   f"Used: {memory_info['percent_used']}%")

        # Adjust batch size and accumulation steps based on device and memory
        if device.type == "mps":
            config.defrost()
            original_batch_size = config.DATA.BATCH_SIZE
            config.DATA.BATCH_SIZE = 1
            config.TRAIN.ACCUMULATION_STEPS = original_batch_size
            config.DATA.IMG_SIZE = 384  # Reduced image size for MPS
            config.DATA.NUM_WORKERS = 0  # No workers for MPS
            config.DATA.PIN_MEMORY = False
            config.freeze()
            logger.info(f"MPS optimized settings - Batch size: {config.DATA.BATCH_SIZE}, "
                       f"Accumulation steps: {config.TRAIN.ACCUMULATION_STEPS}")
        elif memory_info['percent_used'] > 70 and config.DATA.BATCH_SIZE > 1:
            config.defrost()
            original_batch_size = config.DATA.BATCH_SIZE
            config.DATA.BATCH_SIZE = max(1, config.DATA.BATCH_SIZE // 2)
            config.TRAIN.ACCUMULATION_STEPS = original_batch_size // config.DATA.BATCH_SIZE
            config.freeze()
            logger.info(f"Adjusted batch size from {original_batch_size} to {config.DATA.BATCH_SIZE} "
                       f"with gradient accumulation steps = {config.TRAIN.ACCUMULATION_STEPS}")

        return device, config

    except Exception as e:
        raise RuntimeError(f"Error setting up device: {str(e)}")

def setup_model_and_optimizer(config, device, logger):
    """Setup model and optimizer with proper parameter groups"""
    try:
        # Build model
        model = build_model(config, device, logger)

        # Create parameter groups for different learning rates
        param_groups = []

        # Backbone parameters
        backbone_decay = []
        backbone_no_decay = []
        decoder_decay = []
        decoder_no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'backbone' in name:
                if 'norm' in name or 'bias' in name:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
            else:
                if 'norm' in name or 'bias' in name:
                    decoder_no_decay.append(param)
                else:
                    decoder_decay.append(param)

        # Create parameter groups with proper learning rates
        param_groups = [
            {
                'params': backbone_decay,
                'lr': config.TRAIN.BASE_LR * config.TRAIN.BACKBONE_LR_MULTIPLIER,
                'weight_decay': config.TRAIN.WEIGHT_DECAY,
                'name': 'backbone_decay'
            },
            {
                'params': backbone_no_decay,
                'lr': config.TRAIN.BASE_LR * config.TRAIN.BACKBONE_LR_MULTIPLIER,
                'weight_decay': 0.0,
                'name': 'backbone_no_decay'
            },
            {
                'params': decoder_decay,
                'lr': config.TRAIN.BASE_LR,
                'weight_decay': config.TRAIN.WEIGHT_DECAY,
                'name': 'decoder_decay'
            },
            {
                'params': decoder_no_decay,
                'lr': config.TRAIN.BASE_LR,
                'weight_decay': 0.0,
                'name': 'decoder_no_decay'
            }
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        # Create optimizer
        optimizer = build_optimizer(config, param_groups)

        # Log parameter information
        from optimizer import log_optimizer_info
        log_optimizer_info(param_groups, logger)

        return model, optimizer

    except Exception as e:
        logger.error(f"Failed to setup model and optimizer: {str(e)}")
        raise

def setup_training_options(config, device):
    """Setup training optimizations and hyperparameters."""
    try:
        config.defrost()

        # Device-specific optimizations
        if device.type == "cuda":
            config.TRAIN.GRADIENT_CHECKPOINTING = True
            torch.backends.cudnn.benchmark = True
            config.AMP_ENABLE = True
        elif device.type == "mps":
            config.TRAIN.GRADIENT_CHECKPOINTING = False
            torch.backends.mps.deterministic = True
            config.AMP_ENABLE = False
        else:
            config.TRAIN.GRADIENT_CHECKPOINTING = False
            config.AMP_ENABLE = False

        # Training parameters
        if not hasattr(config.TRAIN, 'FREEZE_BACKBONE'):
            config.TRAIN.FREEZE_BACKBONE = True
        if not hasattr(config.TRAIN, 'UNFREEZE_EPOCH'):
            config.TRAIN.UNFREEZE_EPOCH = 5
        if not hasattr(config.TRAIN, 'BACKBONE_LR_MULTIPLIER'):
            config.TRAIN.BACKBONE_LR_MULTIPLIER = 0.1

        # Memory optimization
        memory_info = get_memory_info()
        if memory_info['percent_used'] > 70:
            config.DATA.NUM_WORKERS = min(4, os.cpu_count() or 1)
            config.TRAIN.GRADIENT_CHECKPOINTING = True

        config.freeze()
        return config

    except Exception as e:
        raise RuntimeError(f"Error setting up training options: {str(e)}")


def setup_dataloaders(config, device, logger):
    """Setup data loaders with proper error handling"""
    try:
        # Adjust num_workers based on device
        optimal_workers = min(4, os.cpu_count() or 1) if device.type != "mps" else 0

        # Create datasets
        datasets = {
            'train': CustomSegmentationDataset(
                root_dir=config.DATA.DATA_PATH,
                split='train',
                img_size=config.DATA.IMG_SIZE,
                cache_mode=config.DATA.CACHE_MODE
            ),
            'val': CustomSegmentationDataset(
                root_dir=config.DATA.DATA_PATH,
                split='val',
                img_size=config.DATA.IMG_SIZE,
                cache_mode='no'
            )
        }

        logger.info(f"Dataset sizes - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}")

        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=True,
                num_workers=optimal_workers,
                pin_memory=device.type == "cuda",
                drop_last=True
            ),
            'val': DataLoader(
                datasets['val'],
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=optimal_workers,
                pin_memory=device.type == "cuda"
            )
        }

        return dataloaders

    except Exception as e:
        raise RuntimeError(f"Failed to setup dataloaders: {str(e)}")


def train(args, config, logger):
    """Training process with enhanced memory management"""
    try:
        device, config = setup_device(config, logger)
        setup_distributed(args=args)
        memory_manager = MemoryManager(device)
        config = setup_training_options(config, device)
        dataloaders = setup_dataloaders(config, device, logger)
        model, optimizer = setup_model_and_optimizer(config, device, logger)

        gradient_manager = GradientCheckpointManager(model, device)
        if config.TRAIN.GRADIENT_CHECKPOINTING:
            gradient_manager.enable_checkpointing()
            logger.info("Gradient checkpointing enabled")

        memory_manager.clear_memory()
        n_iter_per_epoch = len(dataloaders['train'])
        optimizer = build_optimizer_with_model(config, model, logger)

        scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)
        criterion = nn.BCEWithLogitsLoss() if config.MODEL.NUM_CLASSES == 1 else nn.CrossEntropyLoss()

        logger.info(f"Using optimizer: {config.TRAIN.OPTIMIZER.NAME}")
        logger.info(f"Using scheduler: {config.TRAIN.LR_SCHEDULER.NAME}")

        use_amp = config.AMP_ENABLE and device.type == 'cuda'
        scaler = NativeScalerWithGradNormCount(device=device) if use_amp else None
        logger.info(f"AMP enabled: {use_amp}")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=config,
            scaler=scaler
        )

        start_epoch = 0
        max_accuracy = 0.0

        if args.resume:
            resume_path = args.resume
        elif config.TRAIN.AUTO_RESUME:
            resume_path = auto_resume_helper(config.OUTPUT)
        else:
            resume_path = None

        if resume_path:
            max_accuracy = load_checkpoint(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                loss_scaler=scaler,
                logger=logger
            )
            start_epoch = config.TRAIN.START_EPOCH
            logger.info(f"Resumed from epoch {start_epoch}")

        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(start_epoch, config.TRAIN.EPOCHS):
            try:
                memory_manager.clear_memory()
                gradient_manager.update_checkpointing_status()

                train_loss, train_miou = trainer.train_epoch(dataloaders['train'], epoch)
                logger.info(
                    f'Epoch [{epoch + 1}/{config.TRAIN.EPOCHS}] '
                    f'Train Loss: {train_loss:.4f} '
                    f'Train mIoU: {train_miou:.4f}'
                )

                if (epoch + 1) % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
                    memory_manager.clear_memory()
                    val_loss, val_miou = trainer.validate(dataloaders['val'], epoch)

                    # Check early stopping after validation
                    should_stop = trainer._check_early_stopping(val_loss)  # Add this line

                    logger.info(
                        f'Epoch [{epoch + 1}/{config.TRAIN.EPOCHS}] '
                        f'Val Loss: {val_loss:.4f} '
                        f'Val mIoU: {val_miou:.4f}'
                    )

                    # Save checkpoint
                    is_best = val_miou > max_accuracy
                    max_accuracy = max(max_accuracy, val_miou)

                    if is_main_process():
                        save_checkpoint(
                            config=config,
                            epoch=epoch,
                            model=model,
                            max_accuracy=max_accuracy,
                            optimizer=optimizer,
                            lr_scheduler=scheduler,
                            loss_scaler=scaler,
                            logger=logger
                        )

                    if should_stop:
                        logger.info("Early stopping triggered. Training terminated.")
                        break

                scheduler.step(epoch)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_manager.clear_memory()
                    logger.warning("Out of memory error, clearing cache and continuing...")
                    continue
                else:
                    logger.error(f"Error during training: {str(e)}")
                    raise

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f'Training completed in {total_time_str}')
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    except Exception as e:
        logger.error(f"Critical error during training: {str(e)}")
        raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('SMT-UPerNet Training and Testing')
    parser.add_argument('--cfg', type=str, required=True,
                       help='path to config file')
    parser.add_argument('--output', type=str, default='work_dirs',
                       help='output directory')
    parser.add_argument('--tag', type=str, default='default',
                       help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                       help='perform evaluation only')
    parser.add_argument('--resume', type=str, default=None,
                       help='resume from specific checkpoint')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,  # Add this line
                       help='number of distributed processes')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')
    parser.add_argument('opts', help="Modify config options by adding 'KEY VALUE' pairs",
                       default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main():
    """Main function with enhanced error handling"""
    try:
        # Parse arguments and get config
        args = parse_args()
        config = get_config(args)

        # Create output directory
        os.makedirs(config.OUTPUT, exist_ok=True)

        # Set random seed
        if args.seed is not None:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        # Create logger
        logger = create_logger(output_dir=config.OUTPUT, name='train')
        logger.info(f"Full config: {config.dump()}")

        if args.eval:
            logger.info("Evaluation mode not implemented yet")
            return
        else:
            train(args, config, logger)

    except Exception as e:
        print(f"Critical error in main: {str(e)}")
        raise


if __name__ == '__main__':
    main()