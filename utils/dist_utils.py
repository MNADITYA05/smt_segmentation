import os
import torch
import torch.distributed as dist
import torch.nn as nn
import math


def setup_distributed(args=None, rank=None, world_size=None):
    """Setup distributed training"""
    if args is not None:
        rank = args.local_rank
        world_size = args.world_size

    if world_size == 1:
        # Single GPU/CPU training, no need for distributed setup
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])

    # Handle different device types
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if world_size > 1:
        # Only initialize process group if we're doing distributed training
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend,
            init_method='tcp://localhost:12355',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()

    return device


def get_world_size():
    """Get world size"""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank of the process"""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if this is the main process"""
    return get_rank() == 0


def reduce_tensor(tensor):
    """
    Reduce tensor across all GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def gather_tensor(tensor):
    """
    Gather tensor from all GPUs
    """
    output_tensors = [tensor.clone() for _ in range(get_world_size())]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


class DistributedSampler(torch.utils.data.Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MPSScaler:
    """Gradient scaler for MPS device"""

    def __init__(self):
        self.scale = torch.tensor(1.0)
        self._scale_seq_ctr = 0

    def scale(self, loss):
        return loss * self.scale

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self._scale_seq_ctr += 1

    def get_scale(self):
        return self.scale.item()

    def set_scale(self, scale):
        self.scale = torch.tensor(scale)

    def state_dict(self):
        return {
            "scale": self.scale.item(),
            "_scale_seq_ctr": self._scale_seq_ctr,
        }

    def load_state_dict(self, state_dict):
        self.scale = torch.tensor(state_dict["scale"])
        self._scale_seq_ctr = state_dict["_scale_seq_ctr"]


def get_grad_norm_(parameters, norm_type=2.0):
    """
    Get gradient norm of parameters
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
    return total_norm


class NativeScalerWithGradNormCount:
    """Gradient scaler with gradient norm counting and MPS support"""
    state_dict_key = "amp_scaler"

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device_type = device.type
        if self.device_type == "mps":
            self._scaler = MPSScaler()
        elif self.device_type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        if self.device_type == "mps":
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                optimizer.step()
                optimizer.zero_grad()
            else:
                norm = None
            return norm
        elif self.device_type == "cuda":
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
            return norm
        else:
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                optimizer.step()
                optimizer.zero_grad()
            else:
                norm = None
            return norm

    def state_dict(self):
        if self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict)


def save_on_master(*args, **kwargs):
    """Save checkpoint only from master process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print