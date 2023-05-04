import torch
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------
#  Misc options
# -----------------------------------------------------------

_C.OUTPUT_DIR = "./runs/resnet18-fpn"
_C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_C.EVAL_ONLY = False

# -----------------------------------------------------------
#  Data
# -----------------------------------------------------------

_C.DATA = CN()

# Num workers
_C.DATA.NUM_WORKERS = 4

# Data directory
_C.DATA.TRAIN_DIR = "../database/WIDER/train"
_C.DATA.VAL_DIR = "../database/WIDER/val"

# Data info
_C.DATA.CATEGORIES = ["face"]
_C.DATA.NUM_CLASSES = len(_C.DATA.CATEGORIES)

# Transform (data augmentation)
_C.DATA.TRANSFORM_HSV_PROB = 0.5
_C.DATA.TRANSFORM_HGAIN = 5
_C.DATA.TRANSFORM_SGAIN = 30
_C.DATA.TRANSFORM_VGAIN = 30
_C.DATA.TRANSFORM_HFLIP_PROB = 0.5
_C.DATA.TRANSFORM_RESIZE_PADDING = True
_C.DATA.TRANSFORM_RESIZE_LETTERBOX = True
_C.DATA.TRANSFORM_AFFINE_DEGREES = 10
_C.DATA.TRANSFORM_AFFINE_TRANSLATE = 0.0
_C.DATA.TRANSFORM_AFFINE_SCALES = 0.0
_C.DATA.TRANSFORM_AFFINE_SHEAR = 5
_C.DATA.TRANSFORM_TO_TENSOR = True
_C.DATA.TRANSFORM_MEAN = (0.0, 0.0, 0.0)
_C.DATA.TRANSFORM_STD = (1.0, 1.0, 1.0)

# -----------------------------------------------------------
#  Model
# -----------------------------------------------------------

_C.MODEL = CN()
# Arch
_C.MODEL.BACKBONE = "resnet-fpn"
_C.MODEL.HEAD = "ct-head"

# ResNet backbone param
_C.MODEL.RESNET_DEPTH = 18
# ResNet-FPN backbone param
_C.MODEL.RESNET_FPN_DEPTH = 18

# CT head param
_C.MODEL.CT_HEAD_MIN_OVERLAP = 0.75
_C.MODEL.CT_HEAD_HM_LOSS_ALPHA = 2.0
_C.MODEL.CT_HEAD_HM_LOSS_GAMMA = 4.0
_C.MODEL.CT_HEAD_HM_LOSS_WEIGHT = 1.0
_C.MODEL.CT_HEAD_OFFSET_LOSS_WEIGHT = 1.0
_C.MODEL.CT_HEAD_WH_LOSS_WEIGHT = 1.0
_C.MODEL.CT_HEAD_TOPK = 100

# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

_C.TRAIN = CN()
# Max epochs
_C.TRAIN.MAX_EPOCHS = 150
# Batch size
_C.TRAIN.BATCH_SIZE = 8
# Lr
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WARMUP_MUL = 0.01
# Resume
_C.TRAIN.RESUME = True
# Whether to drop last when loading data
_C.TRAIN.DROP_LAST = False
# Log interval (iterations)
_C.TRAIN.LOG_INTERVAL = 10
# Eval interval (epochs)
_C.TRAIN.EVAL_INTERVAL = 10
# Input size
_C.TRAIN.INPUT_H = 640
_C.TRAIN.INPUT_W = 640
# Multiscale
_C.TRAIN.MULTISCALE_STEP = 32
_C.TRAIN.MULTISCALE_PERIOD = 10
_C.TRAIN.MULTISCALE_RANGE = 3
_C.TRAIN.MULTISCALE_END_EPOCH = _C.TRAIN.MAX_EPOCHS
# Mosaic
_C.TRAIN.ENABLE_MOSAIC = False
_C.TRAIN.MOSAIC_PROB = 0.0
_C.TRAIN.MOSAIC_END_EPOCH = _C.TRAIN.MAX_EPOCHS - 10

# -----------------------------------------------------------
#  Test(val)
# -----------------------------------------------------------

_C.TEST = CN()
# Input size
_C.TEST.INPUT_H = _C.TRAIN.INPUT_H
_C.TEST.INPUT_W = _C.TRAIN.INPUT_W
# TopK
_C.TEST.TOPK = 100
# NMS
_C.TEST.WITH_NMS = True
_C.TEST.NMS_THRESH = 0.45
# Evaluate
_C.TEST.PER_CLASS_AP = True
_C.TEST.PER_CLASS_AR = True