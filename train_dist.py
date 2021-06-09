import argparse
import os
import yaml 

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

# from model import DQLR, DQLRnn, DQL
from model_pl import DQLRnnDist
from scheduler import CycleScheduler
from dataloader import VideoDataset

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from test_tube import Experiment

from utils import load_config, LitProgressBar

def create_result_dir(cfg, dev=False):
    print(cfg)
    path = cfg['output_dir']
    num_exp = os.listdir(path)
    # print(num_exp)
    curr_dir_id = len(num_exp)
    output_path = os.path.join(path, "{:04d}".format(curr_dir_id+1))
    if not dev:
        try:
            os.mkdir(output_path)
        except:
            pass
        with open(os.path.join(output_path, 'cfg.yaml'), 'w') as yaml_file:
            yaml.dump(cfg, yaml_file, default_flow_style=False)

    return output_path

def get_output_dir(cfg):
    path = cfg['output_dir']
    num_exp = os.listdir(path)
    # print(num_exp)
    curr_dir_id = len(num_exp)
    output_path = os.path.join(path, "{:04d}".format(curr_dir_id))
    return output_path

def main(hparams):
    model_config = load_config(hparams.model_config)
    training_config = load_config(hparams.experiment_config)


    print(model_config)
    print("####################")
    print(training_config)
    trainer = Trainer(gpus=1, num_nodes=3, accelerator='ddp', fast_dev_run=False, 
                        plugins=DDPPlugin(find_unused_parameters=False))
    output_dir = None
    if trainer.global_rank == 0:
        output_dir = create_result_dir(training_config)
    else:
        output_dir = get_output_dir(training_config)
    training_config['sample_path'] = output_dir
    model = DQLRnnDist(model_config, training_config)
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpus', default=None)
    # parser.add_argument('--size', type=int, default=256)
    # parser.add_argument('--epoch', type=int, default=5600)
    # parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--sched', type=str)
    # parser.add_argument('--pretrained', type=str)
    # # parser.add_argument('--ckp', type=str, default="./checkpoint")
    # parser.add_argument('--bs', type=int, default=32)
    # # parser.add_argument('--sample_path')
    # parser.add_argument('--output', type=str, default='./output')
    # parser.add_argument('--alpha', type=float, default=.05)
    # parser.add_argument('--ltriplet', type=float, default=.05)
    # parser.add_argument('--skip', type=int, default=1)
    # parser.add_argument('path', type=str)

    parser.add_argument('--model_config', type=str)
    parser.add_argument('--experiment_config', type=str)
    args = parser.parse_args()

    main(args)