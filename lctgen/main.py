import sys
import os

from matplotlib.pyplot import ginput

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from trafficgen.utils.typedef import *

import argparse
import random

import numpy as np
import torch
import wandb

from lctgen.config.default import Config, get_config
from lctgen.trainer import BaseTrainer as Trainer

import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    trainer = Trainer(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    
    if config.LOGGER == 'wandb':
        wandb.finish()

    return trainer.save_dir

def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts)
    execute_exp(config, run_type)

if __name__ == "__main__":
    main()