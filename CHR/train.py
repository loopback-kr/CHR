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
from dataloading import XrayDataset, read_object_labels_csv
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='CHR Training')
parser.add_argument('--csv_path',               default='./data/trainval.csv', help='path to csv file for split train and valid set')
parser.add_argument('--data_dir',               default='./data/SIXray10', help='path to dataset (e.g. ../data/')
parser.add_argument('-i', '--image_size',       default=224, type=int, help='image size (default: 224)')
parser.add_argument('-j', '--num_workers',      default=16, type=int, help='number of data loading workers')
parser.add_argument('--deterministic',          action='store_true', help='fix randomness for each train')
parser.add_argument('--epochs',                 default=10, type=int, help='number of total epochs to train model')
parser.add_argument('--start_epoch',            default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--train_batch_size', default=128, type=int, help='mini-batch size of train loader (default: 256)') # runme.sh: 320
parser.add_argument('--valid_batch_size',       default=1, type=int, help='mini-batch size of valid loader (default: 1)')
parser.add_argument('--lr', '--learning_rate',  default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum',               default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight_decay',   default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--network_arch',           default='resnet101CHR', type=str, help='model architecture for training or validation')
parser.add_argument('--model_save_dir',         default='./models', type=str, help='path to save checkpoint')
parser.add_argument('--resume_model_path',      default=None, type=str, help='path to resume checkpoint (default: None)')
parser.add_argument('--device',                 default='cuda', type=str, help='device to load model and data')
args = parser.parse_args()

if args.deterministic:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Load csv and split train, valid sets
images_meta = read_object_labels_csv(args.csv_path)

train_images_meta, valid_images_meta = train_test_split(images_meta, test_size=0.2, random_state=0)
# Define dataset
train_dataset = XrayDataset(join(args.data_dir), train_images_meta, transform_mode='train')
valid_dataset = XrayDataset(join(args.data_dir), valid_images_meta, transform_mode='valid') # TODO: valid_images_meta에 대한 성능평가 아직 안해봄

# Create model architecture
if args.network_arch == 'resnet101'
    model = resnet101(num_classes=5, pretrained=True)
elif args.network_arch == 'resnet101CHR'
    model = resnet101_CHR(num_classes=5, pretrained=True)
else:
    raise KeyError

# Define loss function (criterion)
criterion = MultiLabelSoftMarginLoss()

# Define optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd,
)

state = {
    "train_batch_size": args.train_batch_size,
    "valid_batch_size": args.valid_batch_size,
    "image_size": args.image_size,
    "start_epoch": args.start_epoch,
    "max_epochs": args.epochs,
    "resume_model_path": args.resume_model_path,
    "model_save_dir": args.model_save_dir,
    "epoch_step": {20}, # for lr update
    "num_workers": args.num_workers,
    "device": args.device,
}

Engine(state).learning(model, criterion, train_dataset, valid_dataset, optimizer)