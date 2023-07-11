# Python built-in libs for handling filesystems
import csv
from os.path import join
# Datascience packages
import numpy as np
# Dataset manipulation
from PIL import Image
# PyTorch packages
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
# CHR libs
from util import Warp


def read_object_labels_csv(csv_path, header=True):
    images = []
    num_categories = 0
    with open(csv_path) as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0: # skip header row
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
    def __init__(self, data_dir, images_meta, transform_mode='train', classes=["Gun", "Knife", "Wrench", "Pliers", "Scissors"]):
        self.data_dir = data_dir
        self.classes = classes
        self.images_meta = images_meta
        image_norm_mean = [0.485, 0.456, 0.406]
        image_norm_std = [0.229, 0.224, 0.225]
        
        if transform_mode == 'train':
            self.transforms = Compose(
                [
                    Warp(256), # To crop more bigger size is needed
                    RandomHorizontalFlip(),
                    RandomCrop(224),
                    ToTensor(),
                    Normalize(mean=image_norm_mean, std=image_norm_std),
                ]
            )
        elif transform_mode == 'valid':
            self.transforms = Compose(
                [
                    Warp(224),
                    ToTensor(),
                    Normalize(mean=image_norm_mean, std=image_norm_std),
                ]
            )
        else:
            raise NotImplementedError(f'Unknown transform mode: {transform_mode}')
        
    def __getitem__(self, index):
        path, target = self.images_meta[index]
        image = Image.open(join(self.data_dir, path + ".jpg")).convert("RGB")
        image = self.transforms(image)
        return (image, path), target

    def __len__(self):
        return len(self.images_meta)