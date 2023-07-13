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
                labels[labels == -1 ] = 0
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class XrayDataset(Dataset):
    def __init__(self, data_dir, images_meta, transform_mode='train'):
        self.data_dir = data_dir
        self.images_meta = images_meta
        
        if transform_mode == 'train':
            self.transforms = Compose(
                [
                    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif transform_mode == 'train_with_augmentation':
            self.transforms = Compose(
                [
                    Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif transform_mode == 'valid':
            self.transforms = Compose(
                [
                    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise NotImplementedError(f'Unknown transform mode: {transform_mode}')
        
    def __getitem__(self, index):
        path, label = self.images_meta[index]
        image = Image.open(join(self.data_dir, path + ".jpg")).convert("RGB")
        image = self.transforms(image)
        metadata = {
            "UID": path
        }
        return image, label, metadata

    def __len__(self):
        return len(self.images_meta)