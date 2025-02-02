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
from networks import resnet18, resnet101, resnet101_CHR
from datasets import XrayDataset, read_object_labels_csv
from sklearn.model_selection import train_test_split
from util import log


parser = argparse.ArgumentParser(description='CHR Training')
parser.add_argument('--csv_path',               default='./data/trainval.csv', help='path to csv file for split train and valid set')
parser.add_argument('--data_dir',               default='./data/SIXray10', help='path to dataset (e.g. ../data/')
parser.add_argument('-i', '--image_size',       default=224, type=int, help='image size (default: 224)')
parser.add_argument('-j', '--num_workers',      default=16, type=int, help='number of data loading workers')
parser.add_argument('--deterministic',          action='store_true', help='fix randomness for each train')
parser.add_argument('--epochs',                 default=10, type=int, help='number of total epochs to train model')
parser.add_argument('--start_epoch',            default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size',       default=128, type=int, help='mini-batch size of train loader (default: 256)') # runme.sh: 320
parser.add_argument('--lr', '--learning_rate',  default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum',               default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight_decay',   default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--network_arch',           default='resnet101CHR', type=str, help='model architecture for training or validation')
parser.add_argument('--model_save_dir',         default='./models', type=str, help='path to save checkpoint')
parser.add_argument('--resume_model_path',      default=None, type=str, help='path to resume checkpoint (default: None)')
parser.add_argument('--device',                 default='cuda', type=str, help='device to load model and data')
parser.add_argument('--seed',                   default=0, type=int, help='seed number for deterministic')
parser.add_argument('--use_supervision',        default=True, type=bool, help='whether to use supervision of CHR')
parser.add_argument('--use_maskloss',           default=True, type=bool, help='whether to use maskloss of CHR')
parser.add_argument('--use_wandb',              default=True, type=bool, help='whether to use wandb')
parser.add_argument('--wandb_key',              default=os.environ['WANDB_KEY'], type=str, help='whether to use wandb')
parser.add_argument('--wandb_project',          default="CHR", type=str, help='whether to use wandb')
parser.add_argument('--wandb_name',             default=None, type=str, help='whether to use wandb')
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

if args.use_wandb:
    import wandb
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)

state = {
    "batch_size": args.batch_size,
    "network_arch": args.network_arch,
    "image_size": args.image_size,
    "start_epoch": args.start_epoch,
    "max_epochs": args.epochs,
    "resume_model_path": args.resume_model_path,
    "model_save_dir": args.model_save_dir,
    "epoch_step": {20}, # for lr update
    "num_workers": min(os.cpu_count(), args.num_workers),
    "device": args.device,
    "best_score" : 0,
    "use_supervision" : args.use_supervision,
    "use_maskloss" : args.use_maskloss,
    "classes" : ["Gun", "Knife", "Wrench", "Pliers", "Scissors"],
    "use_wandb" : args.use_wandb,
}
log.info(f'State: {state}')

# Read csv and split to train, validation sets
images_meta = read_object_labels_csv(args.csv_path)[:100]
train_images_meta, valid_images_meta = train_test_split(images_meta, test_size=0.2, random_state=0)
# Define dataset
train_dataset = XrayDataset(data_dir=args.data_dir, image_list=train_images_meta, transform_mode='train')
log.info(f'Load train dataset and metadata: classes: {len(state["classes"])}, number of data: {len(train_dataset)}')
valid_dataset = XrayDataset(data_dir=args.data_dir, image_list=valid_images_meta, transform_mode='valid')
log.info(f'Load valid dataset and metadata: classes: {len(state["classes"])}, number of data: {len(valid_dataset)}')

if args.network_arch == 'resnet18':
    model = resnet18(num_classes=len(state["classes"]), pretrained=True)
elif args.network_arch == 'resnet101':
    model = resnet101(num_classes=len(state["classes"]), pretrained=True)
elif args.network_arch == 'resnet101CHR':
    model = resnet101_CHR(num_classes=len(state["classes"]), pretrained=True)
else:
    raise KeyError

log.info(f'Created model architecture: {model.__class__.__name__}')
log.debug(model)
criterion = nn.MultiLabelSoftMarginLoss()
log.info(f'Define loss function: {criterion}')

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd,
)
log.info(f'Define optimizer: {optimizer}')

Engine(state).learning(model, criterion, train_dataset, valid_dataset, optimizer)

if args.use_wandb:
    wandb.finish()