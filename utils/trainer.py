import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import deque
from copy import deepcopy
from utils.metrics import calculate_miou
from utils.visualization import TrainingVisualizer
from logger import create_logger


class ModelEMA:
    """Model Exponential Moving Average"""

    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.ema.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(),
                                      model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)

    def state_dict(self):
        return self.ema.state_dict()


class TrainingState:
    def __init__(self, trainer):
        self.trainer = trainer
        self.history = {
            'lr_changes': [],
            'grad_norms': [],
            'phase_transitions': [],
            'plateau_counts': 0
        }
        self.metrics_history = deque(maxlen=trainer.config.TRAIN.EARLY_STOPPING.METRIC_HISTORY_SIZE)

    def update(self, epoch, batch_idx, metrics):
        # Track learning rate changes
        current_lrs = [group['lr'] for group in self.trainer.optimizer.param_groups]
        if len(self.history['lr_changes']) == 0 or current_lrs != self.history['lr_changes'][-1]:
            self.history['lr_changes'].append(current_lrs)
            self.trainer.logger.info(f"LR change detected at epoch {epoch}, batch {batch_idx}")

        # Track metrics
        self.metrics_history.append(metrics)

        # Track gradient norms if available
        if 'grad_norm' in metrics:
            self.history['grad_norms'].append(metrics['grad_norm'])


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scaler = scaler
        self.best_miou = 0.0

        # Initialize training state
        self.training_state = TrainingState(self)

        # Initialize model EMA if enabled
        self.model_ema = None
        if config.TRAIN.EMA.ENABLED:
            self.model_ema = ModelEMA(model, decay=config.TRAIN.EMA.DECAY)

        # Initialize metrics tracking
        self.loss_history = deque(maxlen=config.TRAIN.EARLY_STOPPING.METRIC_HISTORY_SIZE)
        self.miou_history = deque(maxlen=config.TRAIN.EARLY_STOPPING.METRIC_HISTORY_SIZE)
        self.grad_norm_history = deque(maxlen=config.TRAIN.EARLY_STOPPING.METRIC_HISTORY_SIZE)

        # Store initial learning rates
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        # Setup logger and visualizer
        self.logger = create_logger(output_dir=config.OUTPUT, name='trainer')
        self.visualizer = TrainingVisualizer(config, self.logger)

        # Setup training phases
        self.backbone_frozen = config.TRAIN.FREEZE_BACKBONE
        if self.backbone_frozen:
            self._freeze_backbone()

        self.current_epoch = 0
        self.plateau_count = 0
        self.best_weights = None

        # Setup early stopping
        self.early_stopping_enabled = config.TRAIN.EARLY_STOPPING.ENABLED
        if self.early_stopping_enabled:
            self.early_stopping_patience = config.TRAIN.EARLY_STOPPING.PATIENCE
            self.early_stopping_min_delta = config.TRAIN.EARLY_STOPPING.MIN_DELTA
            self.early_stopping_counter = 0
            self.best_loss = float('inf')
            self.logger.info(f"Early stopping enabled with patience {self.early_stopping_patience}")

    def _freeze_backbone(self):
        """Freeze backbone layers"""
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = False
        self.logger.info(f"Phase 1/{self.config.TRAIN.UNFREEZE_EPOCH}: Backbone frozen, training decoder only")

    def _unfreeze_backbone(self):
        """Unfreeze backbone layers with proper learning rate adjustment"""
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = True

        # Adjust learning rates for backbone parameters
        for param_group in self.optimizer.param_groups:
            if 'backbone' in param_group['name']:
                param_group['lr'] = param_group['lr'] * self.config.TRAIN.BACKBONE_LR_MULTIPLIER

        self.logger.info(f"Phase 2/{self.config.TRAIN.EPOCHS}: Backbone unfrozen, training full model")

    def _verify_optimizer_state(self):
        """Verify optimizer state consistency and parameter groups"""
        try:
            # Check optimizer parameter groups
            for group in self.optimizer.param_groups:
                # Verify initial learning rate exists
                if 'initial_lr' not in group:
                    raise ValueError(f"Parameter group {group.get('name', 'unnamed')} missing initial_lr")

                # Verify learning rate is valid
                if group['lr'] < 0:
                    raise ValueError(f"Invalid learning rate {group['lr']} for group {group.get('name', 'unnamed')}")

                # Verify parameters are properly attached
                for p in group['params']:
                    if not isinstance(p, torch.Tensor):
                        raise ValueError("Parameter is not a torch.Tensor")
                    if not p.is_leaf:
                        raise ValueError("Parameter is not a leaf tensor")
                    if p.requires_grad and p.grad is None:
                        p.grad = torch.zeros_like(p)

                # Verify weight decay is non-negative
                if group['weight_decay'] < 0:
                    raise ValueError(
                        f"Invalid weight decay {group['weight_decay']} for group {group.get('name', 'unnamed')}")

            # Verify optimizer state exists
            if len(self.optimizer.state) == 0 and any(len(g['params']) > 0 for g in self.optimizer.param_groups):
                raise ValueError("Optimizer has empty state but non-empty parameter groups")

            # Verify parameter gradients
            trainable_params = [p for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad]
            if not trainable_params:
                raise ValueError("No trainable parameters found")

            # Log verification success
            self.logger.info("Optimizer state verification completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Optimizer state verification failed: {str(e)}")

            # Log detailed state for debugging
            self.logger.debug("Current optimizer state:")
            for i, group in enumerate(self.optimizer.param_groups):
                self.logger.debug(f"Group {i}: {group.keys()}")

            raise RuntimeError(f"Optimizer state verification failed: {str(e)}")


    def _check_plateau(self):
        """Enhanced plateau detection"""
        if len(self.loss_history) < self.config.TRAIN.EARLY_STOPPING.METRIC_HISTORY_SIZE:
            return False

        recent_losses = list(self.loss_history)
        variance = np.var(recent_losses)
        is_plateau = variance < self.config.TRAIN.EARLY_STOPPING.MIN_DELTA

        if is_plateau:
            self.plateau_count += 1
            self.logger.info(f"Plateau detected (count: {self.plateau_count})")
        return is_plateau

    def _handle_plateau(self):
        """Implement countermeasures for plateaus"""
        if self.plateau_count == 1:
            # First countermeasure: Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            self.logger.info("Learning rate reduced due to plateau")

        elif self.plateau_count == 2:
            # Second countermeasure: Adjust batch normalization momentum
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.momentum = min(m.momentum * 1.2, 0.9)
            self.logger.info("BatchNorm momentum adjusted")

        elif self.plateau_count >= self.config.TRAIN.EARLY_STOPPING.PLATEAU_THRESHOLD:
            # Final countermeasure: Switch to EMA model if available
            if self.model_ema is not None:
                self.model.load_state_dict(self.model_ema.ema.state_dict())
                self.logger.info("Switched to EMA model weights")
                return True

        return False

    def _check_early_stopping(self, val_loss):
        """Enhanced early stopping with countermeasures"""
        if not self.early_stopping_enabled:
            return False

        if val_loss < (self.best_loss - self.early_stopping_min_delta):
            self.best_loss = val_loss
            self.early_stopping_counter = 0
            # Save best weights
            self.best_weights = {
                key: value.cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            return False

        self.early_stopping_counter += 1

        # Implement progressive countermeasures
        if self.early_stopping_counter == self.early_stopping_patience // 2:
            if self.best_weights is not None:
                self.model.load_state_dict(self.best_weights)
                self.logger.info("Restored best model weights")

        if self.early_stopping_counter >= self.early_stopping_patience:
            self.logger.info(f'Early stopping triggered. Best validation loss: {self.best_loss:.6f}')
            return True

        self.logger.info(
            f'Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}'
            f'Current loss: {val_loss:.6f}, Best loss: {self.best_loss:.6f}'
        )
        return False

    def _update_training_phase(self, epoch):
        """Synchronized phase update with state verification"""
        current_phase = "Phase 1 (Decoder)" if self.backbone_frozen else "Phase 2 (Full)"

        if self.backbone_frozen and epoch >= self.config.TRAIN.UNFREEZE_EPOCH:
            self.logger.info(f"Preparing phase transition at epoch {epoch}")

            # Save current state
            pre_transition_state = {
                key: value.cpu().clone()
                for key, value in self.model.state_dict().items()
            }

            try:
                # Unfreeze with state tracking
                self._unfreeze_backbone()
                self.backbone_frozen = False

                # Verify model state
                self._verify_optimizer_state()

            except Exception as e:
                self.logger.error(f"Phase transition failed: {str(e)}")
                # Restore pre-transition state
                self.model.load_state_dict(pre_transition_state)
                raise

            self.logger.info(f"Completed transition from {current_phase} to Phase 2 (Full)")

    def train_epoch(self, train_loader, epoch):
        self.current_epoch = epoch
        self._update_training_phase(epoch)

        # Set model to training mode
        self.model.train()

        total_loss = 0
        total_iou = 0
        total_grad_norm = 0
        num_batches = len(train_loader)

        # Initialize grad_norm for each batch
        grad_norm = 0.0

        phase = "Phase 1 (Decoder)" if epoch < self.config.TRAIN.UNFREEZE_EPOCH else "Phase 2 (Full)"
        pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f'Training Epoch {epoch + 1}')

        for batch_idx, (images, masks) in pbar:
            try:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with autocast
                with torch.amp.autocast(self.device.type if self.device.type != 'mps' else 'cpu',
                                        enabled=self.config.AMP_ENABLE):
                    outputs = self.model(images)
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                                mode='bilinear', align_corners=True)
                    loss = self.criterion(outputs, masks)

                # Backward pass with automatic mixed precision
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.config.TRAIN.CLIP_GRAD:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.TRAIN.CLIP_GRAD
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.TRAIN.CLIP_GRAD
                        )
                    self.optimizer.step()

                # Update EMA model if enabled
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step_update(epoch * len(train_loader) + batch_idx)

                # Calculate metrics
                iou = calculate_miou(outputs.detach(), masks, self.config.MODEL.NUM_CLASSES)
                total_loss += loss.item()
                total_iou += iou
                if grad_norm != 0.0:  # Only add non-zero grad norms
                    total_grad_norm += grad_norm.item()

                # Update progress bar
                pbar.set_description(
                    f'{phase} Epoch: {epoch + 1}/{self.config.TRAIN.EPOCHS} '
                    f'[{batch_idx + 1}/{num_batches}] '
                    f'Loss: {loss.item():.4f} '
                    f'IoU: {iou:.4f} '
                    f'GradNorm: {grad_norm:.4f}'
                )

            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

        # Update metrics history
        self.loss_history.append(avg_loss)
        self.miou_history.append(avg_iou)
        self.grad_norm_history.append(avg_grad_norm)

        # Check for plateau and implement countermeasures
        if self._check_plateau():
            self._handle_plateau()

        # Update metrics and plot
        self.visualizer.update_metrics(
            epoch=epoch,
            train_metrics={
                'loss': avg_loss,
                'miou': avg_iou,
                'grad_norm': avg_grad_norm
            },
            learning_rate=self.optimizer.param_groups[0]['lr']
        )

        return avg_loss, avg_iou

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        if self.model_ema is not None:
            self.model_ema.ema.eval()

        total_loss = 0
        total_iou = 0
        ema_total_loss = 0
        ema_total_iou = 0
        num_batches = len(val_loader)

        phase = "Phase 1 (Decoder)" if epoch < self.config.TRAIN.UNFREEZE_EPOCH else "Phase 2 (Full)"
        pbar = tqdm(enumerate(val_loader), total=num_batches, desc=f'Validating Epoch {epoch + 1}')

        for batch_idx, (images, masks) in pbar:
            try:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Regular model forward pass
                outputs = self.model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                            mode='bilinear', align_corners=True)
                loss = self.criterion(outputs, masks)
                iou = calculate_miou(outputs, masks, self.config.MODEL.NUM_CLASSES)

                # EMA model forward pass if available
                if self.model_ema is not None:
                    ema_outputs = self.model_ema.ema(images)
                    if ema_outputs.shape[-2:] != masks.shape[-2:]:
                        ema_outputs = F.interpolate(ema_outputs, size=masks.shape[-2:],
                                                    mode='bilinear', align_corners=True)
                    ema_loss = self.criterion(ema_outputs, masks)
                    ema_iou = calculate_miou(ema_outputs, masks, self.config.MODEL.NUM_CLASSES)

                    ema_total_loss += ema_loss.item()
                    ema_total_iou += ema_iou

                total_loss += loss.item()
                total_iou += iou

                pbar.set_description(
                    f'{phase} Val Epoch: {epoch + 1}/{self.config.TRAIN.EPOCHS} '
                    f'[{batch_idx + 1}/{num_batches}] '
                    f'Loss: {loss.item():.4f} '
                    f'IoU: {iou:.4f}'
                )

            except Exception as e:
                self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                raise

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        # Check EMA performance
        if self.model_ema is not None:
            ema_avg_loss = ema_total_loss / num_batches
            ema_avg_iou = ema_total_iou / num_batches

            if ema_avg_loss < avg_loss:
                self.logger.info("EMA model performing better, switching to EMA weights")
                self.model.load_state_dict(self.model_ema.ema.state_dict())
                avg_loss = ema_avg_loss
                avg_iou = ema_avg_iou

        # Update validation metrics and plot
        self.visualizer.update_metrics(
            epoch=epoch,
            val_metrics={'loss': avg_loss, 'miou': avg_iou}
        )

        # Generate plots after both training and validation are complete
        self.visualizer.plot_metrics(current_phase=phase)

        return avg_loss, avg_iou