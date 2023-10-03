from typing import List, Optional, Union
import yacs.config
from .path_cfg import _C as _C_PATH

class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
# task config can be a list of conifgs like "A.yaml,B.yaml"
_C.SEED = 0
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "default"

_C.EXPERIMENT_DIR = "results/debug"
_C.EXPERIMENT_NAME = "pipeline"
_C.TENSORBOARD_DIR = "tensorboard"
_C.WANDB_PROJ = "test"
_C.GPU = None # GPU id (-1 if use CPU)
_C.SAVE_CHECKPOINT = False

# Number of model updates during training
_C.MAX_EPOCHES = 15
_C.CHECKPOINT_EPOCHES = 15

_C.CHECKPOINT_INTERVAL = 1
_C.VAL_INTERVAL = 5
_C.VIS_INTERVAL = 10
_C.LOG_INTERVAL_STEPS = 1
_C.LIMIT_VAL_BATCHES = 1.0
_C.LOAD_CHECKPOINT_MODEL = False
_C.LOAD_CHECKPOINT_TRAINER = False
_C.LOAD_CHECKPOINT_PATH = None
_C.DEBUG = False

# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL_NAME = "lctgen"

# -----------------------------------------------------------------------------
# TRAIN CONFIG
# -----------------------------------------------------------------------------

_C.TRAIN = CN()
# The split to train on
_C.TRAIN.SPLIT = "train"
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.LR = 0.0003
_C.TRAIN.OPTIMIZER = 'AdamW' # [Adam, AdamW, SDG]
_C.TRAIN.WEIGHT_DECAY = 0.0004
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = True

_C.TRAIN.LR_MUL_MODEL = None
# _C.TRAIN.LR_MUL_MODEL = ['GPT2']
_C.TRAIN.LR_MUL_FACTOR = 0.1

_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.TYPE = 'MultiStepLR'
_C.TRAIN.SCHEDULER.GAMMA = 0.1
_C.TRAIN.SCHEDULER.STEP = 5
_C.TRAIN.SCHEDULER.ETA_MIN = 0.0
_C.TRAIN.SCHEDULER.WARMUP_EPOCHES = 10
_C.TRAIN.SCHEDULER.MILESTONES = [20, 25]

_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True
_C.TRAIN.NUM_WORKERS = 1

# -----------------------------------------------------------------------------
# VAL CONFIG
# -----------------------------------------------------------------------------
_C.VAL = CN()
# The split to validate on
_C.VAL.SPLIT = "test"
_C.VAL.BATCH_SIZE = 16
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False
_C.VAL.NUM_WORKERS = 1
_C.VAL.COMPUTE_LOSS = True

# -----------------------------------------------------------------------------
# TEST CONFIG
# -----------------------------------------------------------------------------
_C.TEST = CN()
# The split to test on
_C.TEST.SPLIT = "test"
_C.TEST.BATCH_SIZE = 16
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False
_C.TEST.NUM_WORKERS = 1
_C.TEST.COMPUTE_LOSS = True

# -----------------------------------------------------------------------------
# DATASET CONFIG
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = 'WaymoOpenMotion'
_C.DATASET.CACHE = True
_C.DATASET.DATA_USAGE = 8
_C.DATASET.DATA_LIST = CN()

_C.DATASET.DATA_LIST.TRAIN = 'mini_train.txt'
_C.DATASET.DATA_LIST.VAL = 'mini_val.txt'
_C.DATASET.DATA_LIST.TEST = 'mini_val.txt'

_C.DATASET.REF_PATH = None
_C.DATASET.RANGE = 50
_C.DATASET.MAX_AGENT_NUM = 32
_C.DATASET.THRES = 5
_C.DATASET.MAX_TIME_STEP = 190
_C.DATASET.TIME_SAMPLE_GAP = 190

_C.DATASET.AUGMENTATION = CN()
_C.DATASET.AUGMENTATION.ENABLE = False
_C.DATASET.AUGMENTATION.RANDOM_MASK_RATE = 0.15
_C.DATASET.AUGMENTATION.RANDOM_EMPTY_RATE = 0.15

_C.DATASET.TEXT = CN()
_C.DATASET.TEXT.CONTEXT_LENGTH = 77
_C.DATASET.TEXT.TYPE = 'static'

_C.DATASET.TRAJ_TYPE = 'xy_theta_relative'
_C.DATASET.ENABLE_OUT_RANGE_TRAJ = False

_C.DATASET.TEXT.ALL_CLASSES = ['cnt', 'pos', 'distance', 'paral_direction', 'vertical_horizontal', 'speed', 'ego_speed', 'cnt_bins', 'cnt_norm', 'lane_cnt']
_C.DATASET.TEXT.CLASS = ['pos']
_C.DATASET.TEXT.SAMPLE_NUMS = [1]
_C.DATASET.TEXT.RANDOM_SAMPLE = False
_C.DATASET.TEXT.RANDOM_SHUFFLE = False
_C.DATASET.TEXT.NEGATIVE_SAMPLE_NUM = 9
_C.DATASET.TEXT.COUNT_BASE = 1
_C.DATASET.TEXT.DISTANCE_BASE = 5
_C.DATASET.TEXT.SPEED_BASE = 2.5
_C.DATASET.TEXT.USE_PADDING = True
_C.DATASET.TEXT.PADDING = -1
_C.DATASET.TEXT.FLATTEN = True
_C.DATASET.TEXT.USE_TRAJ = False
_C.DATASET.TEXT.ACTION_STEP = 4
_C.DATASET.TEXT.ACTION_DIM = 1 # 1 for direction/acceleration; 2 for dimention + acceleration

_C.DATASET.IMAGE = CN()
_C.DATASET.IMAGE.RENDER = False
_C.DATASET.INCLUDE_LANE_INFO = False

# -----------------------------------------------------------------------------
# LOSS CONFIG
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# loss_type: ['null', 'clip', 'cosine]
_C.LOSS.TYPE = 'clip'
_C.LOSS.CLIP = CN()
_C.LOSS.CLIP.LOGIT_SCALE = 1.0
_C.LOSS.WEIGHTS = [1.0]
_C.LOSS.AE = CN()
_C.LOSS.AE.TEXT_AE = False
_C.LOSS.DETR = CN()
_C.LOSS.DETR.MATCH_METHOD = 'hungarian' # ['hungarian', 'sequential']
_C.LOSS.DETR.PRED_BACKGROUND = True
_C.LOSS.DETR.MATCH_COST = CN()
_C.LOSS.DETR.MATCH_COST.CLASS = 1.0
_C.LOSS.DETR.AE_MODES = ['input'] # ['input', 'text', 'ref_text']
_C.LOSS.DETR.LOSSES = ['labels']
# _C.LOSS.DETR.LOSSES = ['labels', 'attributes', 'vae']
_C.LOSS.DETR.WEIGHT = CN()
_C.LOSS.DETR.WEIGHT.labels = 1.0
_C.LOSS.DETR.WEIGHT.attributes = 1.0
_C.LOSS.DETR.WEIGHT.vae = 1.0
_C.LOSS.DETR.ATTR_WEIGHT = CN()
_C.LOSS.DETR.ATTR_WEIGHT.speed = 0.1
_C.LOSS.DETR.ATTR_WEIGHT.pos = 1.0
_C.LOSS.DETR.ATTR_WEIGHT.vel_heading = 1.0
_C.LOSS.DETR.ATTR_WEIGHT.bbox = 1.0
_C.LOSS.DETR.ATTR_WEIGHT.heading = 1.0
_C.LOSS.DETR.ATTR_WEIGHT.motion = 1.0
_C.LOSS.DETR.EOS_COEF = 0.1
_C.LOSS.DETR.USE_CENTER_MASK = False
_C.LOSS.DETR.TEXT_AE = False
_C.LOSS.DETR.ALIGNMENT = CN()
_C.LOSS.DETR.ALIGNMENT.ENABLE = False
_C.LOSS.DETR.ALIGNMENT.WEIGHT = CN()
_C.LOSS.DETR.ALIGNMENT.WEIGHT.CLIP = 1.0
_C.LOSS.DETR.ALIGNMENT.WEIGHT.COSINE = 1.0
_C.LOSS.DETR.ALIGNMENT.WEIGHT.MSE = 1.0

# -----------------------------------------------------------------------------
# METRIC CONFIG
# -----------------------------------------------------------------------------
_C.METRIC = CN()
_C.METRIC.TYPE = ['R_topk']
_C.METRIC.TOPK = 1
_C.METRIC.MMD = CN()
_C.METRIC.MMD.KERNEL_MUL = 1.0
_C.METRIC.MMD.KERNEL_NUM = 1
_C.METRIC.MMD.ATTR = ['position', 'speed', 'size', 'heading']

# -----------------------------------------------------------------------------
# LLM CONFIG
# -----------------------------------------------------------------------------
_C.LLM = CN()
_C.LLM.TYPE = 'codex'
_C.LLM.CODEX = CN()
_C.LLM.CODEX.MODEL = 'gpt-3.5-turbo'
_C.LLM.CODEX.TEMPERATURE = 0.0
_C.LLM.CODEX.MAX_TOKENS = 2048
_C.LLM.CODEX.BEST_OF = 1
_C.LLM.CODEX.PROMPT_FILE = 'chatapi.prompt'
_C.LLM.CODEX.SYS_PROMPT_FILE = None

# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'lctgen'

_C.MODEL.PREDICT_EGO = False
_C.MODEL.USE_REL_HEADING = True

_C.MODEL.MOTION = CN()
_C.MODEL.MOTION.ENABLE = False
_C.MODEL.MOTION.ENCODE_MOTION = False
_C.MODEL.MOTION.STEP = 4
_C.MODEL.MOTION.CUMSUM = False
# mlp, mlp_gmm, mtf
_C.MODEL.MOTION.PRED_MODE = 'mlp'
_C.MODEL.MOTION.PRED_HEADING_VEL = False
_C.MODEL.MOTION.K = 6
_C.MODEL.MOTION.CLS_WEIGHT = 0.1

_C.MODEL.SCENE = CN()

_C.MODEL.SCENE.REMOVE_COLLISION = True

_C.MODEL.SCENE.DEBUG = CN()
_C.MODEL.SCENE.DEBUG.RANDOM_ENCODE = False
_C.MODEL.SCENE.DEBUG.ZERO_ENCODE = False


_C.MODEL.SCENE.INIT_CFG = CN()

_C.MODEL.SCENE.INIT_CFG.hidden_dim = 1024
_C.MODEL.SCENE.INIT_CFG.max_num = 32
_C.MODEL.SCENE.INIT_CFG.gaussian_comp = 5
_C.MODEL.SCENE.INIT_CFG.context_num = 32

_C.MODEL.SCENE.INIT_CFG.ENCODER = CN()
_C.MODEL.SCENE.INIT_CFG.ENCODER.TYPE = 'mcg' #'transformer'
_C.MODEL.SCENE.INIT_CFG.ENCODER.NHEAD = 4
_C.MODEL.SCENE.INIT_CFG.ENCODER.NLAYER = 2
_C.MODEL.SCENE.INIT_CFG.ENCODER.DROPOUT = 0.1
_C.MODEL.SCENE.INIT_CFG.ENCODER.FF_DIM = 1024
_C.MODEL.SCENE.INIT_CFG.ENCODER.ACTIVATION = 'gelu'

_C.MODEL.SCENE.INIT_CFG.DECODER = CN()
_C.MODEL.SCENE.INIT_CFG.DECODER.TYPE = 'maskformer' # detr, maskformer, z_decode, agent_decode
_C.MODEL.SCENE.INIT_CFG.DECODER.MAP_POS  = 'none' # 'sine', 'learned', 'none'
_C.MODEL.SCENE.INIT_CFG.DECODER.QUERY_POS  = 'none' # 'sine', 'learned', 'none'
_C.MODEL.SCENE.INIT_CFG.DECODER.QUERY_NUM = 32
_C.MODEL.SCENE.INIT_CFG.DECODER.NHEAD = 4
_C.MODEL.SCENE.INIT_CFG.DECODER.NLAYER = 2
_C.MODEL.SCENE.INIT_CFG.DECODER.DROPOUT = 0.1
_C.MODEL.SCENE.INIT_CFG.DECODER.FF_DIM = 2048
_C.MODEL.SCENE.INIT_CFG.DECODER.ACTIVATION = 'gelu'
_C.MODEL.SCENE.INIT_CFG.DECODER.LANE_NUM = 384
_C.MODEL.SCENE.INIT_CFG.DECODER.MLP_DIM = 512
_C.MODEL.SCENE.INIT_CFG.DECODER.ATTR_GMM_ENABLE = False
_C.MODEL.SCENE.INIT_CFG.DECODER.ATTR_GMM_K = 5

_C.MODEL.SCENE.INIT_CFG.ATTR_QUERY = CN()
_C.MODEL.SCENE.INIT_CFG.ATTR_QUERY.USE_LEARNABLE_QUERY = False
_C.MODEL.SCENE.INIT_CFG.ATTR_QUERY.POS_ENCODING_DIM = 1024

_C.MODEL.SCENE.HIDDEN_DIMS = [2048, 1024, 512]
_C.MODEL.SCENE.EMBEDDING_DIM = 512
_C.MODEL.SCENE.AGGREGATION = 'max'
_C.MODEL.SCENE.NORMALIZE = False

_C.merge_from_other_cfg(_C_PATH)

_C.register_renamed_key

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    # config.FINETUNE = finetune_config.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config