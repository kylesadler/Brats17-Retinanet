from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from train import test
parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--skip-test",
    dest="skip_test",
    help="Do not test the final model",
    action="store_true",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

model.load_state_dict(torch.load("10_22_19/model_0179999.pth")['model'])
model.eval()

test(cfg, model, args.distributed)
