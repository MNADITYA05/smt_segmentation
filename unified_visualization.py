import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from data.build import build_loader
from models.build import build_model
from config import get_config
from logger import create_logger
from utils.checkpoint import load_checkpoint
from utils.dist_utils import setup_distributed


class UnifiedVisualizationHelper:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else 'cpu')

        # Initialize storage for activations and gradients
        self._cached_activation = None
        self._cached_gradients = None
        self._hooks = []

        # Initialize model and data
        self.model = self._setup_model()
        self.data_loader = self._setup_dataloader()

        # Register hooks for GradCAM
        self._register_hooks()

    def _setup_model(self):
        """Setup and load model with proper error handling"""
        try:
            model = build_model(self.config, self.device, self.logger)
            model.eval()

            if self.config.MODEL.RESUME:
                load_checkpoint(
                    config=self.config,
                    model=model,
                    optimizer=None,
                    lr_scheduler=None,
                    loss_scaler=None,
                    logger=self.logger
                )
            return model
        except Exception as e:
            self.logger.error(f"Error setting up model: {str(e)}")
            raise

    def _setup_dataloader(self):
        """Setup data loader with proper error handling"""
        try:
            _, _, _, val_loader = build_loader(self.config)
            self.logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
            return val_loader
        except Exception as e:
            self.logger.error(f"Error setting up data loader: {str(e)}")
            raise

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer"""

        def forward_hook(module, input, output):
            self._cached_activation = output

        def backward_hook(module, grad_input, grad_output):
            self._cached_gradients = grad_output[0]

        # Find target layer (decode_head.conv_seg)
        target_found = False
        for name, module in self.model.named_modules():
            if name == 'decode_head.conv_seg':
                self._hooks.extend([
                    module.register_forward_hook(forward_hook),
                    module.register_full_backward_hook(backward_hook)
                ])
                target_found = True
                break

        if not target_found:
            raise ValueError("Could not find target layer 'decode_head.conv_seg'")

    def denormalize_image(self, tensor):
        """Denormalize image tensor for visualization"""
        try:
            dataset = self.data_loader.dataset
            mean = dataset.normalize_config['mean']
            std = dataset.normalize_config['std']

            mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
            std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

            return torch.clamp(tensor * std + mean, 0, 1)
        except Exception as e:
            self.logger.error(f"Error in denormalization: {str(e)}")
            raise

    def compute_gradcam(self, image_tensor):
        """Compute GradCAM with improved focus on target regions"""
        try:
            # Create a copy of the image tensor that requires gradients
            image_tensor = image_tensor.clone()
            image_tensor.requires_grad_(True)

            # Forward pass
            self.model.zero_grad()
            output = self.model(image_tensor.unsqueeze(0))

            if isinstance(output, (tuple, list)):
                output = output[-1]

            # For binary segmentation with improved target selection
            if output.shape[1] == 1:
                prob = torch.sigmoid(output)

                # Use gradient scaling for better focus
                scale = torch.ones_like(prob)
                scale[prob > 0.5] = 2.0  # Enhance focus on high probability regions

                target = prob * scale
            else:
                prob = F.softmax(output, dim=1)
                target = prob[:, 1].unsqueeze(1)

            # Compute gradients with respect to target regions
            target.sum().backward()

            # Get gradients and features
            gradients = self._cached_gradients
            features = self._cached_activation

            # Improved gradient weighting
            weights = F.adaptive_avg_pool2d(gradients, 1)

            # Apply spatial attention
            spatial_weights = F.softmax(gradients.sum(dim=1, keepdim=True), dim=2)
            weights = weights * spatial_weights

            # Weight features with attention
            cam = (features * weights).sum(dim=1, keepdim=True)

            # Apply ReLU for positive contributions
            cam = F.relu(cam)

            # Resize to input size
            cam = F.interpolate(
                cam,
                size=image_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # Enhanced normalization
            if cam.max() != cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

                # Apply contrast enhancement
                cam = torch.pow(cam, 0.7)  # Adjust gamma for better contrast
                threshold = 0.3  # Adjust threshold to control focus
                cam[cam < threshold] = 0

            # Convert to numpy array
            cam = cam.squeeze().detach().cpu().numpy()

            return cam

        except Exception as e:
            self.logger.error(f"Error computing GradCAM: {str(e)}")
            raise

    def visualize_unified(self, image, mask, prediction, gradcam, save_path):
        """Create comprehensive visualization including input, mask, prediction, and improved GradCAM"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            plt.subplots_adjust(wspace=0.1, hspace=0)

            # Original image
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            # Ground truth mask
            axes[1].imshow(mask.cpu().numpy(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # Predicted mask
            axes[2].imshow(prediction.cpu().numpy(), cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            # GradCAM heatmap
            im = axes[3].imshow(gradcam, cmap='jet')
            axes[3].set_title('GradCAM')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

            # GradCAM overlay
            axes[4].imshow(image_np, cmap='gray')
            im = axes[4].imshow(gradcam, cmap='jet', alpha=0.6)
            axes[4].set_title('GradCAM Overlay')
            axes[4].axis('off')
            plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

            # Save with high quality
            plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error in unified visualization: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)

    def run_visualization(self, save_dir):
        """Run unified visualization pipeline"""
        os.makedirs(save_dir, exist_ok=True)

        for idx, (images, masks) in enumerate(tqdm(self.data_loader)):
            if idx >= self.config.VIS.MAX_SAMPLES:
                break

            try:
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Get predictions (without gradients)
                with torch.no_grad():
                    outputs = self.model(images)

                    # Ensure correct size
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=masks.shape[-2:],
                            mode='bilinear',
                            align_corners=self.config.MODEL.DECODE_HEAD.ALIGN_CORNERS
                        )

                    # Get binary prediction
                    predictions = (torch.sigmoid(outputs) > 0.5).float() if self.config.MODEL.NUM_CLASSES == 1 else \
                        outputs.argmax(dim=1)

                # Compute GradCAM
                gradcam = self.compute_gradcam(images[0])

                # Save unified visualization
                base_path = os.path.join(save_dir, f'sample_{idx}_unified.png')
                self.visualize_unified(
                    self.denormalize_image(images[0]),
                    masks[0],
                    predictions[0] if self.config.MODEL.NUM_CLASSES == 1 else predictions[0].float(),
                    gradcam,
                    base_path
                )

            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {str(e)}")
                continue

    def __del__(self):
        """Clean up hooks when object is deleted"""
        for hook in self._hooks:
            hook.remove()


def main():
    """Main function with enhanced error handling"""
    logger = None
    try:
        parser = argparse.ArgumentParser('SMT-UPerNet Unified Visualization')
        parser.add_argument('--cfg', type=str, required=True, help='path to config file')
        parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('--output', type=str, default='unified_visualizations', help='output directory')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
        parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs",
                            default=None, nargs=argparse.REMAINDER)
        args = parser.parse_args()

        # Create output directory
        os.makedirs(args.output, exist_ok=True)

        # Setup logger
        logger = create_logger(output_dir=args.output, name='unified_visualization')
        logger.info("Starting unified visualization process")

        # Get and prepare config
        config = get_config(args)
        config.defrost()
        config.DATA.BATCH_SIZE = 1
        config.DATA.NUM_WORKERS = 0
        config.DATA.CACHE_MODE = 'no'
        config.EVAL_MODE = True
        if args.checkpoint:
            config.MODEL.RESUME = args.checkpoint
        config.freeze()

        # Setup distributed
        setup_distributed(args)

        # Create visualization helper and run
        viz_helper = UnifiedVisualizationHelper(config, logger)
        viz_dir = os.path.join(config.OUTPUT, 'unified_visualizations')
        viz_helper.run_visualization(viz_dir)

        logger.info(f"Visualizations saved to {viz_dir}")

    except Exception as e:
        if logger:
            logger.error(f"Visualization failed: {str(e)}")
        else:
            print(f"Error before logger initialization: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()