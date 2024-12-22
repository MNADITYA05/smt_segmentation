import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# Data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 2  # Kept small for segmentation
_C.DATA.DATA_PATH = 'datasets'
_C.DATA.DATASET = 'custom'
_C.DATA.IMG_SIZE = 512
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 4  # Increased from 2
_C.DATA.CACHE_MODE = 'part'  # Changed to 'part' for better memory usage
_C.DATA.NORMALIZE = CN()
_C.DATA.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
_C.DATA.NORMALIZE.STD = [0.229, 0.224, 0.225]

# Model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'smt'
_C.MODEL.NAME = 'smt_small_224'
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.2
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.CONVERT_WEIGHTS = True

# In config.py, under MODEL section
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NUM_LAYERS = 4  # Default number of layers for SMT

# SMT parameters
_C.MODEL.SMT = CN()
_C.MODEL.SMT.PATCH_SIZE = 4
_C.MODEL.SMT.IN_CHANS = 3
_C.MODEL.SMT.EMBED_DIMS = [64, 128, 256, 512]
_C.MODEL.SMT.CA_NUM_HEADS = [4, 4, 4, -1]
_C.MODEL.SMT.SA_NUM_HEADS = [-1, -1, 8, 16]
_C.MODEL.SMT.MLP_RATIOS = [4, 4, 4, 2]
_C.MODEL.SMT.QKV_BIAS = True
_C.MODEL.SMT.QK_SCALE = None
_C.MODEL.SMT.USE_LAYERSCALE = True
_C.MODEL.SMT.LAYERSCALE_VALUE = 1e-4
_C.MODEL.SMT.DROP_RATE = 0.0
_C.MODEL.SMT.ATTN_DROP_RATE = 0.0
_C.MODEL.SMT.DROP_PATH_RATE = 0.2
_C.MODEL.SMT.DEPTHS = [3, 4, 18, 2]
_C.MODEL.SMT.CA_ATTENTIONS = [1, 1, 1, 0]
_C.MODEL.SMT.HEAD_CONV = 3
_C.MODEL.SMT.EXPAND_RATIO = 2

# Decode head settings
_C.MODEL.DECODE_HEAD = CN()
_C.MODEL.DECODE_HEAD.TYPE = 'UPerHead'
_C.MODEL.DECODE_HEAD.IN_CHANNELS = [64, 128, 256, 512]
_C.MODEL.DECODE_HEAD.IN_INDEX = [0, 1, 2, 3]
_C.MODEL.DECODE_HEAD.POOL_SCALES = [1, 2, 3, 6]
_C.MODEL.DECODE_HEAD.CHANNELS = 512
_C.MODEL.DECODE_HEAD.DROPOUT_RATIO = 0.1
_C.MODEL.DECODE_HEAD.NUM_CLASSES = 2
_C.MODEL.DECODE_HEAD.ALIGN_CORNERS = False
_C.MODEL.DECODE_HEAD.LOSS_DECODE = CN()
_C.MODEL.DECODE_HEAD.LOSS_DECODE.TYPE = 'CrossEntropyLoss'
_C.MODEL.DECODE_HEAD.LOSS_DECODE.USE_SIGMOID = True
_C.MODEL.DECODE_HEAD.LOSS_DECODE.LOSS_WEIGHT = 1.0
_C.MODEL.DECODE_HEAD.NORM_CFG = CN()
_C.MODEL.DECODE_HEAD.NORM_CFG.TYPE = 'SyncBN'
_C.MODEL.DECODE_HEAD.NORM_CFG.REQUIRES_GRAD = True

# Training settings
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 6e-5
_C.TRAIN.WARMUP_LR = 1e-6
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.GRADIENT_CHECKPOINTING = False

# In config.py (add under TRAIN section)
_C.TRAIN.EARLY_STOPPING = CN()
_C.TRAIN.EARLY_STOPPING.ENABLED = True
_C.TRAIN.EARLY_STOPPING.PATIENCE = 7
_C.TRAIN.EARLY_STOPPING.MIN_DELTA = 1e-4

# Training strategy
_C.TRAIN.FREEZE_BACKBONE = True
_C.TRAIN.UNFREEZE_EPOCH = 5
_C.TRAIN.BACKBONE_LR_MULTIPLIER = 0.1
_C.TRAIN.LAYER_DECAY = 0.75

# Add under _C.TRAIN
_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLED = True
_C.TRAIN.EMA.DECAY = 0.9999

# Modify existing early stopping section
_C.TRAIN.EARLY_STOPPING.PLATEAU_THRESHOLD = 3
_C.TRAIN.EARLY_STOPPING.METRIC_HISTORY_SIZE = 5


# Optimizer settings
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.AMSGRAD = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'  # Options: 'cosine', 'linear', 'step', 'multistep'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.NOISE_PCT = 0.67  # Fixed typo here
_C.TRAIN.LR_SCHEDULER.NOISE_STD = 1.0
_C.TRAIN.LR_SCHEDULER.NOISE_SEED = 42   # Fixed typo here

_C.EVAL_MODE = False

# Augmentation settings
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.2
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# Testing settings
_C.TEST = CN()
_C.TEST.CROP = True
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False
_C.TEST.MODE = 'whole'

# Visualization settings
_C.VIS = CN()
_C.VIS.MAX_SAMPLES = 10
_C.VIS.FEATURE_VIS = True
_C.VIS.SAVE_PATH = 'visualizations'
_C.VIS.DPI = 300

# Add under _C.VIS
_C.VIS.PLOT_INTERVAL = 1       # Plot every N epochs
_C.VIS.PLOTS_DIR = 'plots'     # Directory for saving plots
_C.VIS.SAVE_FORMAT = 'png'     # Plot save format
_C.VIS.PLOT_DPI = 300         # Plot quality settings

# Runtime settings
_C.AMP_ENABLE = True
_C.OUTPUT = 'work_dirs'
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 25
_C.SEED = 0
_C.LOCAL_RANK = 0
_C.THROUGHPUT_MODE = False
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    try:
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg_dir = os.path.dirname(cfg_file)

        # Handle base configs
        if yaml_cfg.get('BASE', None):
            for base_cfg in yaml_cfg.get('BASE'):
                if base_cfg:
                    base_cfg_path = os.path.join('configs', base_cfg)
                    _update_config_from_file(config, base_cfg_path)

        print('=> merge config from {}'.format(cfg_file))
        config.merge_from_file(cfg_file)
        config.freeze()
    except Exception as e:
        print(f"Error loading config file: {cfg_file}")
        print(f"Error message: {str(e)}")
        raise e


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # Handle output directory
    output_dir = args.output if hasattr(args, 'output') and args.output else 'work_dirs'

    # Handle tag
    tag = args.tag if hasattr(args, 'tag') and args.tag else 'default'

    # Set output path
    config.OUTPUT = os.path.join(output_dir, config.MODEL.NAME, tag)

    # Handle resume path
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)
    return config