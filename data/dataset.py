import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, cache_mode="no", normalize_config=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.cache_mode = cache_mode

        # Set up paths
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')

        # Validate directory structure
        self._validate_directories()

        # Get and validate image files
        self.images = self._get_validated_files()

        # Initialize cache
        self.samples_cache = {}
        if self.cache_mode != "no":
            self.init_cache()

        # Set up normalization
        self.normalize_config = normalize_config or {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

        # Set up transforms
        self.transform = self.get_transforms()

    def _validate_directories(self):
        """Validate the dataset directory structure"""
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

    def _get_validated_files(self):
        """Get and validate image files with corresponding masks"""
        images = sorted(os.listdir(self.images_dir))
        valid_images = []
        print(f"Validating {len(images)} images for {self.split} set...")

        for img_name in images:
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)

            try:
                # Check if mask exists
                if not os.path.exists(mask_path):
                    print(f"Warning: Missing mask for {img_name}")
                    continue

                # Verify image can be opened
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Corrupted image {img_name}")
                    continue

                # Verify mask can be opened
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Corrupted mask for {img_name}")
                    continue

                # Verify image and mask dimensions match
                if img.shape[:2] != mask.shape[:2]:
                    print(f"Warning: Dimension mismatch for {img_name}")
                    continue

                # Verify mask can be binarized
                try:
                    mask = (mask > 127).astype(np.uint8)
                    valid_images.append(img_name)
                except Exception as e:
                    print(f"Warning: Cannot binarize mask for {img_name}: {e}")
                    continue

            except Exception as e:
                print(f"Error validating {img_name}: {str(e)}")
                continue

        if not valid_images:
            raise ValueError(f"No valid image-mask pairs found in the {self.split} dataset")

        print(f"Found {len(valid_images)} valid image-mask pairs for {self.split} set")
        return valid_images

    def _load_image(self, path):
        """Load and validate image"""
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Failed to load image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            return None

    def _load_mask(self, path):
        """Load and preprocess mask"""
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Failed to load mask")

            # Binarize the mask
            mask = (mask > 127).astype(np.uint8)
            return mask
        except Exception as e:
            print(f"Error loading mask {path}: {str(e)}")
            return None

    def get_transforms(self):
        """Get transforms for the dataset"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=self.normalize_config['mean'],
                std=self.normalize_config['std']
            ),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        """Get item with enhanced error handling"""
        try:
            img_name = self.images[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)

            if idx in self.samples_cache:
                image, mask = self.samples_cache[idx]
            else:
                image = self._load_image(img_path)
                mask = self._load_mask(mask_path)

                if image is None or mask is None:
                    raise ValueError("Failed to load image or mask")

                if self.cache_mode == "part":
                    self.samples_cache[idx] = (image, mask)

            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask'].long()

        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            # Return a valid but empty sample as fallback
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(self.img_size, self.img_size,
                                                                             dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def init_cache(self):
        """Initialize cache for faster data loading"""
        if self.cache_mode == "full":
            print("Initializing full cache...")
            for idx in range(len(self)):
                img_name = self.images[idx]
                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.masks_dir, img_name)

                image = self._load_image(img_path)
                mask = self._load_mask(mask_path)

                if image is not None and mask is not None:
                    self.samples_cache[idx] = (image, mask)