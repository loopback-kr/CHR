# Python built-in libs for handling filesystems
import sys, os, json, pickle, csv, re, random, logging, importlib, argparse
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
from pathlib import Path
from shutil import copy, copytree, rmtree
from copy import deepcopy
from glob import glob, iglob
# Datascience packages for medical imaging
import numpy as np
# Dataset manipulation
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
from util import Warp


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print("[dataset] read", file)
    with open(file, "r") as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1 : num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class XrayDataset(Dataset):
    def __init__(self, data_dir, images_meta, transform_mode='train'):
        self.classes = ["Gun", "Knife", "Wrench", "Pliers", "Scissors"]

        self.data_dir = data_dir
        
        # 이 값들은 많은 Vision 모델들의 pretraining에 사용된 ImageNet 데이터셋의 학습 시에 얻어낸 값들이다. ImageNet 데이터셋은 질 좋은 이미지들을 다량 포함하고 있기에 이런 데이터셋에서 얻어낸 값이라면 어떤 이미지 데이터 셋에서도 잘 작동할 것이라는 가정하에 이 값들을 기본 값으로 세팅해 놓은 것이다.
        image_normalization_mean = [0.485, 0.456, 0.406]
        image_normalization_std = [0.229, 0.224, 0.225]
        
        if transform_mode == 'train':
            self.transforms = Compose(
                [
                    Warp(256), # Crop하기 위해서 조금 키워서 resize
                    RandomHorizontalFlip(),
                    RandomCrop(224),
                    ToTensor(),
                    Normalize(mean=image_normalization_mean, std=image_normalization_std),
                ]
            )
        elif transform_mode == 'valid':
            self.transforms = Compose(
                [
                    Warp(224),
                    ToTensor(),
                    Normalize(mean=image_normalization_mean, std=image_normalization_std),
                ]
            )
        else:
            raise KeyError
        
        self.images_meta = images_meta

        print(
            "[dataset] X-ray classification set=%s number of classes=%d  number of images=%d"
            % (set, len(self.classes), len(self.images_meta))
        )

    def __getitem__(self, index):
        path, target = self.images_meta[index]
        img = Image.open(join(self.data_dir, path + ".jpg")).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, path), target

    def __len__(self):
        return len(self.images_meta)

    def get_number_classes(self):
        return len(self.classes)
