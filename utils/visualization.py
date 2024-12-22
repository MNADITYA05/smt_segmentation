import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import numpy as np
from logger import create_logger
import torch
import gc
from termcolor import colored


class TrainingVisualizer:
    def __init__(self, config, logger=None):
        """Initialize training visualizer"""
        self.config = config
        self.logger = logger or create_logger(output_dir=config.OUTPUT, name='visualizer')

        # Create directories
        self.metrics_dir = os.path.join(config.OUTPUT, 'metrics')
        self.plots_dir = os.path.join(config.VIS.SAVE_PATH, 'training')
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize metrics with empty lists
        self.metrics = {
            'train_loss': [],
            'train_miou': [],
            'val_loss': [],
            'val_miou': [],
            'learning_rates': [],
            'epochs': []
        }

        self.current_phase = "Phase 1 (Decoder)"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set plot style
        plt.style.use('seaborn')
        self.load_metrics()

        self.logger.info(f"Initialized training visualizer. Plots will be saved to {self.plots_dir}")

    def _ensure_equal_lengths(self):
        """Ensure all metric lists have equal lengths by padding with last values"""
        if not self.metrics['epochs']:
            return

        max_len = len(self.metrics['epochs'])
        for key in ['train_loss', 'train_miou', 'val_loss', 'val_miou', 'learning_rates']:
            while len(self.metrics[key]) < max_len:
                pad_value = self.metrics[key][-1] if self.metrics[key] else 0
                self.metrics[key].append(pad_value)

    def clear_memory(self):
        """Clear GPU memory and matplotlib figures"""
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_latest_metrics(self):
        """Get the latest metrics safely"""
        if not self.metrics['epochs']:
            return None

        return {
            'train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else 0,
            'train_miou': self.metrics['train_miou'][-1] if self.metrics['train_miou'] else 0,
            'val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else 0,
            'val_miou': self.metrics['val_miou'][-1] if self.metrics['val_miou'] else 0
        }

    def update_metrics(self, epoch, train_metrics=None, val_metrics=None, learning_rate=None):
        """Update training metrics with synchronization"""
        try:
            # Add epoch if not present
            if epoch not in self.metrics['epochs']:
                self.metrics['epochs'].append(epoch)

            current_idx = self.metrics['epochs'].index(epoch)

            # Update train metrics if provided
            if train_metrics:
                while len(self.metrics['train_loss']) <= current_idx:
                    self.metrics['train_loss'].append(
                        self.metrics['train_loss'][-1] if self.metrics['train_loss'] else 0)
                while len(self.metrics['train_miou']) <= current_idx:
                    self.metrics['train_miou'].append(
                        self.metrics['train_miou'][-1] if self.metrics['train_miou'] else 0)

                self.metrics['train_loss'][current_idx] = train_metrics['loss']
                self.metrics['train_miou'][current_idx] = train_metrics['miou']

            # Update validation metrics if provided
            if val_metrics:
                while len(self.metrics['val_loss']) <= current_idx:
                    self.metrics['val_loss'].append(
                        self.metrics['val_loss'][-1] if self.metrics['val_loss'] else 0)
                while len(self.metrics['val_miou']) <= current_idx:
                    self.metrics['val_miou'].append(
                        self.metrics['val_miou'][-1] if self.metrics['val_miou'] else 0)

                self.metrics['val_loss'][current_idx] = val_metrics['loss']
                self.metrics['val_miou'][current_idx] = val_metrics['miou']

            # Update learning rate if provided
            if learning_rate is not None:
                while len(self.metrics['learning_rates']) <= current_idx:
                    self.metrics['learning_rates'].append(
                        self.metrics['learning_rates'][-1] if self.metrics['learning_rates'] else learning_rate)
                self.metrics['learning_rates'][current_idx] = learning_rate

            # Ensure all lists are synchronized
            self._ensure_equal_lengths()
            self.save_metrics()

        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            raise

    def save_metrics(self):
        """Save metrics to JSON with error handling"""
        try:
            metrics_file = os.path.join(self.metrics_dir, 'training_metrics.json')
            formatted_metrics = {
                'epochs': self.metrics['epochs'],
                'training': {
                    'loss': self.metrics['train_loss'],
                    'miou': self.metrics['train_miou']
                },
                'validation': {
                    'loss': self.metrics['val_loss'],
                    'miou': self.metrics['val_miou']
                },
                'learning_rates': self.metrics['learning_rates'],
                'current_phase': self.current_phase
            }

            with open(metrics_file, 'w') as f:
                json.dump(formatted_metrics, f, indent=4)

            self.logger.debug(f"Metrics saved to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

    def load_metrics(self):
        """Load existing metrics with error handling"""
        metrics_file = os.path.join(self.metrics_dir, 'training_metrics.json')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics['epochs'] = data.get('epochs', [])
                    self.metrics['train_loss'] = data.get('training', {}).get('loss', [])
                    self.metrics['train_miou'] = data.get('training', {}).get('miou', [])
                    self.metrics['val_loss'] = data.get('validation', {}).get('loss', [])
                    self.metrics['val_miou'] = data.get('validation', {}).get('miou', [])
                    self.metrics['learning_rates'] = data.get('learning_rates', [])
                    self.current_phase = data.get('current_phase', "Phase 1 (Decoder)")
                # Ensure loaded metrics are synchronized
                self._ensure_equal_lengths()
                self.logger.info(f"Loaded existing metrics from {metrics_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load existing metrics: {str(e)}")

    def plot_metrics(self, current_phase=None):
        """Generate and save plots with synchronized data"""
        try:
            if not self.metrics['epochs']:
                self.logger.warning("No data to plot yet")
                return

            self.clear_memory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure data is synchronized before plotting
            self._ensure_equal_lengths()

            # Create figure and subplots
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3)

            # Plot Loss
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.metrics['epochs'], self.metrics['train_loss'],
                     'b-', label='Training Loss', linewidth=2)
            if any(v != 0 for v in self.metrics['val_loss']):
                ax1.plot(self.metrics['epochs'], self.metrics['val_loss'],
                         'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot mIoU
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.metrics['epochs'], self.metrics['train_miou'],
                     'b-', label='Training mIoU', linewidth=2)
            if any(v != 0 for v in self.metrics['val_miou']):
                ax2.plot(self.metrics['epochs'], self.metrics['val_miou'],
                         'r-', label='Validation mIoU', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mIoU')
            ax2.set_title('Training and Validation mIoU')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot Learning Rate
            ax3 = fig.add_subplot(gs[1, :])
            if self.metrics['learning_rates']:
                ax3.plot(self.metrics['epochs'], self.metrics['learning_rates'],
                         'g-', label='Learning Rate', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.set_title('Learning Rate Schedule')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)
                ax3.legend()

            # Update phase and set title
            if current_phase:
                self.current_phase = current_phase
            fig.suptitle(f'Training Metrics - {self.current_phase}', fontsize=14, y=1.02)

            # Save plot
            plot_file = os.path.join(
                self.plots_dir,
                f'training_metrics_epoch_{max(self.metrics["epochs"])}_{timestamp}.png'
            )
            plt.savefig(plot_file, dpi=self.config.VIS.DPI, bbox_inches='tight')
            plt.close()

            self.clear_memory()
            self.logger.info(colored(f'Saved training plots to {plot_file}', 'green'))
            return plot_file

        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise

    def get_current_metrics(self):
        """Get latest metrics with error handling"""
        if not self.metrics['epochs']:
            return None

        latest = {
            'epoch': self.metrics['epochs'][-1],
            'train_loss': self.metrics['train_loss'][-1],
            'train_miou': self.metrics['train_miou'][-1],
        }

        if self.metrics['val_loss']:
            latest.update({
                'val_loss': self.metrics['val_loss'][-1],
                'val_miou': self.metrics['val_miou'][-1]
            })

        return latest

    def state_dict(self):
        """Get state for checkpointing"""
        return {
            'metrics': self.metrics,
            'current_phase': self.current_phase
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        try:
            self.metrics = state_dict['metrics']
            self.current_phase = state_dict['current_phase']
            self._ensure_equal_lengths()  # Ensure loaded state is synchronized
            self.logger.info("Loaded visualization state from checkpoint")
        except Exception as e:
            self.logger.error(f"Error loading visualization state: {str(e)}")
            raise

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'train_loss': [],
            'train_miou': [],
            'val_loss': [],
            'val_miou': [],
            'learning_rates': [],
            'epochs': []
        }
        self.logger.info("Reset all metrics")