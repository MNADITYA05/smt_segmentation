import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from enum import Enum
from typing import Optional, Tuple, Union


class LayerType(Enum):
    """Enumeration of supported layer types for GradCAM visualization."""
    CONV_SEG = 1
    FPN_BOTTLENECK = 2
    PSP_MODULE = 3
    ATTENTION = 4


class SegGradCAM:
    """
    GradCAM implementation optimized for semantic segmentation models.
    Supports binary and multi-class segmentation with various backbone architectures.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            target_layer_name: str,
            device: torch.device,
            input_size: Tuple[int, int] = (224, 224),
            use_cuda: bool = False
    ):
        """
        Initialize SegGradCAM.

        Args:
            model: The segmentation model to analyze
            target_layer_name: Name of the target layer for visualization
            device: Device to run computations on
            input_size: Input image size (height, width)
            use_cuda: Whether to use CUDA for acceleration
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = device
        self.input_size = input_size
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Initialize storage for activations and gradients
        self._cached_activation = None
        self._cached_gradients = None
        self._hooks = []

        # Determine layer type and register hooks
        self.layer_type = self._determine_layer_type()
        self._register_hooks()

        # Set model to evaluation mode but enable gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

    def _determine_layer_type(self) -> LayerType:
        """
        Determine the type of the target layer.

        Returns:
            LayerType: The determined layer type
        """
        if 'conv_seg' in self.target_layer_name:
            return LayerType.CONV_SEG
        elif 'fpn_bottleneck' in self.target_layer_name:
            return LayerType.FPN_BOTTLENECK
        elif 'psp' in self.target_layer_name:
            return LayerType.PSP_MODULE
        elif 'attention' in self.target_layer_name:
            return LayerType.ATTENTION
        else:
            raise ValueError(f"Unsupported layer type: {self.target_layer_name}")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self._cached_activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._cached_gradients = grad_output[0].detach()

        # Find and validate target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found in model")

        # Register hooks
        self._hooks.extend([
            target_layer.register_forward_hook(forward_hook),
            target_layer.register_full_backward_hook(backward_hook)
        ])

    def _preprocess_image(
            self,
            image: np.ndarray,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image (numpy array)
            mean: Normalization mean values
            std: Normalization standard deviation values

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")

        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1:
            image = image / 255.0

        # Handle different input formats
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] not in [3, 4]:
            raise ValueError("Input image must be RGB or RGBA")
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Remove alpha channel

        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Normalize
        for t, m, s in zip(image, mean, std):
            t.sub_(m).div_(s)

        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(self.device)

        # Resize if needed
        if image.shape[-2:] != self.input_size:
            image = F.interpolate(
                image,
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )

        return image

    def _compute_cam(
            self,
            gradients: torch.Tensor,
            activations: torch.Tensor,
            target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute Class Activation Map.

        Args:
            gradients: Gradient tensor
            activations: Activation tensor
            target_size: Desired output size

        Returns:
            np.ndarray: Computed CAM
        """
        # Ensure proper shapes
        if gradients.ndim != 4 or activations.ndim != 4:
            raise ValueError("Gradients and activations must be 4D tensors")

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weight activations by importance
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        # Resize to target size
        cam = F.interpolate(
            cam,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Normalize properly with epsilon for numerical stability
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_min == cam_max:
            return np.zeros(target_size)

        cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)

        return cam.squeeze().cpu().numpy()

    def generate_cam(
            self,
            image: np.ndarray,
            target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.

        Args:
            image: Input image
            target_size: Desired output size (defaults to input image size)

        Returns:
            np.ndarray: Generated CAM
        """
        if target_size is None:
            target_size = image.shape[:2]

        # Clear any existing gradients
        self.model.zero_grad()

        # Preprocess and forward pass
        input_tensor = self._preprocess_image(image)
        outputs = self.model(input_tensor)

        # Handle hierarchical outputs
        if isinstance(outputs, (tuple, list)):
            seg_output = outputs[-1]  # Take final segmentation output
        else:
            seg_output = outputs

        # Handle binary vs multi-class segmentation
        if seg_output.shape[1] == 1:  # Binary
            target = seg_output
            target_activation = torch.sigmoid(target)
        else:  # Multi-class
            target = seg_output[:, 1].unsqueeze(1)  # Keep dimensions consistent
            target_activation = F.softmax(seg_output, dim=1)[:, 1]

        # Compute gradients
        target.backward(gradient=torch.ones_like(target))

        # Ensure we have gradients and activations
        if self._cached_gradients is None or self._cached_activation is None:
            raise RuntimeError("Failed to capture gradients or activations")

        # Compute CAM
        cam = self._compute_cam(
            self._cached_gradients,
            self._cached_activation,
            target_size
        )

        return cam

    def visualize(
            self,
            image: np.ndarray,
            save_path: str,
            alpha: float = 0.5,
            colormap: int = cv2.COLORMAP_JET
    ) -> None:
        """
        Generate and save visualization.

        Args:
            image: Input image
            save_path: Path to save visualization
            alpha: Overlay transparency
            colormap: OpenCV colormap to use for visualization
        """
        # Generate CAM
        cam = self.generate_cam(image)

        # Create heatmap
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8),
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = (1 - alpha) * image + alpha * heatmap / 255.0
        overlay = np.clip(overlay, 0, 1)

        # Create visualization
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # GradCAM heatmap
        plt.subplot(132)
        plt.imshow(cam, cmap='jet')
        plt.title(f'GradCAM: {self.target_layer_name}')
        plt.axis('off')

        # Overlay
        plt.subplot(133)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def __del__(self):
        """Clean up hooks when object is deleted."""
        for hook in self._hooks:
            hook.remove()