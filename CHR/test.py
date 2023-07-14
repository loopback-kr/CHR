# Python built-in libs for handling filesystems
import sys, os, json, pickle, csv, re, random, logging, importlib, argparse
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
from pathlib import Path
from shutil import copy, copytree, rmtree
from copy import deepcopy
from glob import glob, iglob
# PyTorch packages
import torch, torch.nn as nn, numpy as np
# CHR libs
from engine import Engine
from networks import resnet101, resnet101_CHR
from loss import MultiLabelSoftMarginLoss
from datasets import XrayDataset, read_object_labels_csv
from util import log


parser = argparse.ArgumentParser(description='CHR Training')
parser.add_argument('--csv_path',               default='./data/test.csv', help='path to csv file for split train and valid set')
parser.add_argument('--data_dir',               default='./data/SIXray10', help='path to dataset (e.g. ../data/')
parser.add_argument('-i', '--image_size',       default=224, type=int,  help='image size (default: 224)')
parser.add_argument('-j', '--num_workers',      default=16, type=int,  help='number of data loading workers')
parser.add_argument('--deterministic',          action='store_true', help='fix randomness for each train')
parser.add_argument('--batch_size',             default=128, type=int,  help='mini-batch size of valid loader (default: 1)')
parser.add_argument('--network_arch',           default='resnet101CHR', type=str, help='model architecture for training or validation')
parser.add_argument('--model_path',             default='./models.bak/model_best_85.7037.pth', type=str,  help='path to latest checkpoint (default: none)')
parser.add_argument('--device',                 default='cuda', type=str, help='device to load model and data')
parser.add_argument('--use_supervision',        default=True, type=bool, help='whether to use supervision of CHR')
parser.add_argument('--seed',                   default=0, type=int, help='seed number for deterministic')
args = parser.parse_args()

if args.deterministic:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f'Set deterministic seed: {args.seed}')
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

state = {
    "valid_batch_size": args.batch_size,
    "image_size": args.image_size,
    "num_workers": min(os.cpu_count(), args.num_workers),
    "device": args.device,
    "use_supervision" : args.use_supervision,
    "classes" : ["Gun", "Knife", "Wrench", "Pliers", "Scissors"],
}
log.info(f'State: {state}')

# Define dataset
valid_dataset = XrayDataset(join(args.data_dir), read_object_labels_csv(args.csv_path), transform_mode='valid')

# Load model
if args.network_arch == 'resnet101':
    model = resnet101(num_classes=len(state['classes']), pretrained=False)
elif args.network_arch == 'resnet101CHR':
    model = resnet101_CHR(num_classes=len(state['classes']), pretrained=False)
else: raise ModuleNotFoundError

if isfile(args.model_path):
    log.info(f"Loading checkpoint {args.model_path}")
    checkpoint = torch.load(args.model_path)
    state["start_epoch"] = checkpoint["epoch"]
    state["best_score"] = checkpoint["best_score"]
    model.load_state_dict(checkpoint["state_dict"])
    log.info(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
    criterion = nn.MultiLabelSoftMarginLoss()
else:
    raise FileNotFoundError(f"No checkpoint found at {args.model_path}")