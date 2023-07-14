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


def read_object_labels_csv(csv_path: str, header: bool = True) -> list:
    image_list = []
    with open(csv_path) as f:
        for i, row in enumerate(csv.reader(f)):
            if header and i == 0:
                continue
            else:
                uid = row[0]
                labels = np.array(row[1:6], dtype=np.float32)
                labels[labels == -1] = 0
                labels = torch.from_numpy(labels)
                image_list.append((uid, labels))
    return image_list

class XrayDataset(Dataset):
    def __init__(self, data_dir, image_list, transform_mode='train'):
        self.data_dir = data_dir
        self.image_list = image_list
        
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
        uid, label = self.image_list[index]
        image = Image.open(join(self.data_dir, f"{uid}.jpg")).convert("RGB")
        image = self.transforms(image)
        metadata = {
            "UID": uid
        }
        return image, label, metadata

    def __len__(self):
        return len(self.image_list)