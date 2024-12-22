import os
import torch
from logger import create_logger


def load_checkpoint(config, model, optimizer=None, lr_scheduler=None, loss_scaler=None, logger=None):
    """
    Load checkpoint from a file.
    Args:
        config (CN): Config object.
        model (Module): Model to load checkpoint.
        optimizer (Optimizer, optional): Optimizer to load checkpoint.
        lr_scheduler (LRScheduler, optional): Learning rate scheduler to load checkpoint.
        loss_scaler (GradScaler, optional): Gradient scaler to load checkpoint.
        logger (Logger, optional): Logger to log info.
    Returns:
        max_accuracy (float): Maximum accuracy achieved so far.
    """
    if logger is None:
        logger = create_logger(output_dir=config.OUTPUT, name='checkpoint')

    # Decide whether to load pretrained or resume checkpoint
    if hasattr(config.MODEL, 'RESUME') and config.MODEL.RESUME:
        checkpoint_path = config.MODEL.RESUME
        logger.info(f"==============> Resuming from checkpoint: {checkpoint_path}")
    elif hasattr(config.MODEL, 'PRETRAINED') and config.MODEL.PRETRAINED:
        checkpoint_path = config.MODEL.PRETRAINED
        logger.info(f"==============> Loading pretrained from: {checkpoint_path}")
    else:
        logger.info("==============> No checkpoint provided, starting from scratch")
        return 0.0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove module. prefix if model was saved with DataParallel/DistributedDataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # For fine-tuning, remove classification head if it exists
    state_dict = {k: v for k, v in state_dict.items()
                 if not k.startswith('head.') and not k.startswith('fc.')}

    # Load model state
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loading state dict message: {msg}")

    # Only load optimizer and scheduler states if resuming training
    max_accuracy = 0.0
    if hasattr(config.MODEL, 'RESUME') and config.MODEL.RESUME:
        if not config.EVAL_MODE and optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Loading optimizer succeeded")

            if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("Loading lr_scheduler succeeded")

            if loss_scaler is not None and 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                logger.info("Loading loss_scaler succeeded")

            if 'epoch' in checkpoint:
                config.defrost()
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                config.freeze()
                logger.info(f"Starting epoch: {checkpoint['epoch'] + 1}")

            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']
                logger.info(f"Max accuracy from checkpoint: {max_accuracy:.2f}")

    # Clean up memory
    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    """
    Save checkpoint to a file.
    Args:
        config (CN): Config object.
        epoch (int): Current epoch.
        model (Module): Model to save.
        max_accuracy (float): Maximum accuracy achieved so far.
        optimizer (Optimizer): Optimizer to save.
        lr_scheduler (LRScheduler): Learning rate scheduler to save.
        loss_scaler (GradScaler): Gradient scaler to save.
        logger (Logger): Logger to log info.
    """
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'scaler': loss_scaler.state_dict() if loss_scaler else None,
        'max_accuracy': max_accuracy,
        'epoch': epoch,
        'config': config.dump()
    }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving...")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

    if max_accuracy > getattr(save_checkpoint, 'best_acc', 0.0):
        save_checkpoint.best_acc = max_accuracy
        best_path = os.path.join(config.OUTPUT, 'model_best.pth')
        torch.save(save_state, best_path)
        logger.info(f"New best model saved to {best_path}")


def auto_resume_helper(output_dir):
    """Automatically find the latest checkpoint in output_dir."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]

    if len(checkpoints) == 0:
        return None

    # Get latest checkpoint
    latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints],
                          key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def get_checkpoint_path(config, epoch=None, is_best=False):
    """
    Get path to checkpoint file.
    Args:
        config (CN): Config object.
        epoch (int, optional): Epoch number.
        is_best (bool): Whether to return path to best model.
    Returns:
        str: Path to checkpoint file.
    """
    if is_best:
        return os.path.join(config.OUTPUT, 'model_best.pth')
    if epoch is None:
        return auto_resume_helper(config)
    return os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')