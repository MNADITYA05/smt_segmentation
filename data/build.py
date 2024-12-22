import torch
import torch.distributed as dist
from .dataset import CustomSegmentationDataset

def build_loader(config):
    # Get world size for distributed training
    num_tasks = dist.get_world_size() if dist.is_initialized() else 1
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    # Create datasets with caching support
    dataset_train = CustomSegmentationDataset(
        root_dir=config.DATA.DATA_PATH,
        split='train',
        img_size=config.DATA.IMG_SIZE,
        cache_mode=config.DATA.CACHE_MODE,
        normalize_config={
            'mean': config.DATA.NORMALIZE.MEAN,
            'std': config.DATA.NORMALIZE.STD
        }
    )

    dataset_val = CustomSegmentationDataset(
        root_dir=config.DATA.DATA_PATH,
        split='val',
        img_size=config.DATA.IMG_SIZE,
        cache_mode='no',
        normalize_config={
            'mean': config.DATA.NORMALIZE.MEAN,
            'std': config.DATA.NORMALIZE.STD
        }
    )

    # Create samplers for distributed training
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    ) if num_tasks > 1 else torch.utils.data.RandomSampler(dataset_train)

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, shuffle=False
    ) if num_tasks > 1 else torch.utils.data.SequentialSampler(dataset_val)

    # Create dataloaders with optimized settings
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        prefetch_factor=2 if config.DATA.NUM_WORKERS > 0 else None,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        prefetch_factor=2 if config.DATA.NUM_WORKERS > 0 else None,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val