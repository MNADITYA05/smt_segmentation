import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=512, normalize_config=None):
    """Get training transforms with configurable normalization"""
    if normalize_config is None:
        normalize_config = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ], p=0.3),
        A.Normalize(
            mean=normalize_config['mean'],
            std=normalize_config['std']
        ),
        ToTensorV2()
    ])


def get_val_transforms(img_size=512, normalize_config=None):
    """Get validation transforms with configurable normalization"""
    if normalize_config is None:
        normalize_config = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=normalize_config['mean'],
            std=normalize_config['std']
        ),
        ToTensorV2()
    ])