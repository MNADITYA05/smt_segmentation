import torch
import functools
import psutil


class MemoryManager:
    def __init__(self, device):
        self.device = device
        self.is_mps = device.type == 'mps'

    def clear_memory(self):
        """Clear unused memory"""
        if self.is_mps:
            torch.mps.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()

    def get_memory_status(self):
        """Get current memory status"""
        vm = psutil.virtual_memory()
        return {
            'total': vm.total,
            'available': vm.available,
            'percent': vm.percent,
            'device_type': self.device.type
        }

    def check_memory_threshold(self, threshold=0.9):
        """Check if memory usage is above threshold"""
        vm = psutil.virtual_memory()
        return vm.percent > (threshold * 100)


def checkpoint_wrapper(module_class):
    """Wrapper to enable gradient checkpointing for a module"""
    original_forward = module_class.forward

    @functools.wraps(original_forward)
    def forward_with_checkpoint(self, *args, **kwargs):
        def custom_forward(*inputs):
            return original_forward(self, *inputs, **kwargs)

        if self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                *args,
                preserve_rng_state=True
            )
        else:
            return original_forward(self, *args, **kwargs)

    module_class.forward = forward_with_checkpoint
    return module_class


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for model"""
    if hasattr(model, 'gradient_checkpointing'):
        model.gradient_checkpointing = True
    return model


class GradientCheckpointManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.checkpointing_enabled = False
        self.memory_manager = MemoryManager(device)

    def update_checkpointing_status(self):
        """Update gradient checkpointing based on memory usage"""
        if self.memory_manager.check_memory_threshold(threshold=0.8):
            if not self.checkpointing_enabled:
                self.enable_checkpointing()
        elif self.checkpointing_enabled:
            self.disable_checkpointing()

    def enable_checkpointing(self):
        """Enable gradient checkpointing"""
        self.model = enable_gradient_checkpointing(self.model)
        self.checkpointing_enabled = True

    def disable_checkpointing(self):
        """Disable gradient checkpointing"""
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
        self.checkpointing_enabled = False