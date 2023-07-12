# Python built-in libs for handling filesystems
import sys, os, json, pickle, csv, re, random, logging, importlib, argparse, time
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
from pathlib import Path
from shutil import copy, copytree, copyfile, rmtree
from copy import deepcopy
from glob import glob, iglob
# PyTorch packages
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
import numpy as np
from tqdm import tqdm
from util import log
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn.metrics import average_precision_score

class Engine:
    def __init__(self, state={}):
        self.state = state

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer):
        
        # Data loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.state["train_batch_size"],
            shuffle=True,
            num_workers=self.state["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False
        )
        valid_loader = DataLoader(
            val_dataset,
            batch_size=self.state["valid_batch_size"],
            shuffle=False,
            num_workers=self.state["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Optionally resume from a checkpoint
        if self.state["resume_model_path"] is not None:
            if isfile(self.state["resume_model_path"]):
                log.info(f"Loading checkpoint {self.state['resume_model_path']}")
                checkpoint = torch.load(self.state["resume_model_path"])
                self.state["start_epoch"] = checkpoint["epoch"]
                self.state["best_score"] = checkpoint["best_score"]
                model.load_state_dict(checkpoint["state_dict"])
                log.info(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
            else:
                log.warn(f"No checkpoint found at {self.state['resume_model_path']}")
        
        device = self.state['device']
        model = model.to(device)
        criterion = criterion.to(device)
        log.info(f'Upload model at device:{device}')

        for current_epoch in range(self.state["start_epoch"], self.state["max_epochs"] + 1):
            self.state["epoch"] = current_epoch
            self.update_lr(optimizer)

            # Train for one epoch
            self.train(model, train_loader, criterion, optimizer, epoch=current_epoch, device=device)

            # Evaluate on validation set
            prec1 = self.validate(valid_loader, model, criterion, device=device)

            # Remember best prec@1 and save checkpoint
            if prec1 > self.state["best_score"]:
                self.state["best_score"] = max(prec1, self.state["best_score"])
                os.makedirs(self.state["model_save_dir"], exist_ok=True)
                torch.save({
                        "epoch": current_epoch + 1,
                        "best_score": self.state["best_score"],
                        "state_dict": model.state_dict(),
                    }, join(self.state["model_save_dir"], f"{self.state['network_arch']}_{current_epoch:04d}_{self.state['best_score']}.pth"))
                log.info(f"Saved best model: {self.state['best_score']:.4f}")
        
        log.info(
            f'--------------------------------------------------------------------------------',
            f'Training finished.',
            f'Final epoch: {current_epoch}',
            f'Best score: {self.state["best_score"]}',
        )
    
    def inference(self, model, criterion, test_dataset):
        self.state["best_score"] = 0

        # Data loading
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.state["valid_batch_size"],
            shuffle=False,
            num_workers=self.state["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False
        )

        device = self.state['device']
        model = model.to(device)
        criterion = criterion.to(device)

        self.validate(test_loader, model, criterion, device=device)

        log.info(
            f'--------------------------------------------------------------------------------\n'
            f'Inference finished.\n'
        )

    def train(self, model, data_loader, criterion, optimizer, epoch, device:str='cpu'):

        model.train()
        data_loader = tqdm(data_loader, desc="Train", colour='green', dynamic_ncols=True, leave=False)

        total_loss = []
        mAP = MultilabelAveragePrecision(num_labels=len(self.state['classes']), average="macro", thresholds=None)

        for _, (image, label, metadata) in enumerate(data_loader):

            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output = model(image)
            loss = 0

            for i in range(3 if self.state["use_supervision"] else 1):
                output0 = output[i]
                n_data = np.ones(output0.shape)
                index = np.where(output0.detach().cpu().numpy() < -20)
                if self.state["use_maskloss"] and len(index[0] != 0):
                    n_data[index] = 0
                    n_data = 1 - n_data
                    n_data = (torch.from_numpy(n_data).float().to(device))
                    n_target = 1 - label.cpu().numpy()
                    n_target = torch.from_numpy(n_target).to(device)
                    mask = 1 - torch.mul(n_data, n_target)
                    loss += torch.mean(
                        torch.mul(mask, criterion(output[i], label))
                    )
                else:
                    loss += torch.mean(criterion(output[i], label))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and mAP
            total_loss.append(loss.item())
            mAP.update(output[0].detach().cpu() if self.state['use_supervision'] else output.item(), label.detach().cpu().int())
            data_loader.set_description(desc=f"Train :: Epoch:{epoch:4d}/{self.state['max_epochs']:d}, Loss:{loss.item():.04f}")

        map = mAP.compute()
        mean_total_loss = np.array(total_loss).mean()
        log.info(
            f"Train :: Epoch:{epoch:4d}/{self.state['max_epochs']:d}, Loss:{mean_total_loss:.4f}, mAP:{map:.4f}, Time:{data_loader.format_dict['elapsed']:.4f}"
        )

    def validate(self, data_loader, model, criterion, device:str='cpu'):
        model.eval()
        with torch.no_grad():
            total_loss = []
            mAP = MultilabelAveragePrecision(num_labels=len(self.state['classes']), average="macro", thresholds=None)

            data_loader = tqdm(data_loader, desc='Validating', colour='blue', dynamic_ncols=True, leave=False)
            for _, (image, label, metadata) in enumerate(data_loader):

                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                output = model(image)
                output = output[0] if self.state['use_supervision'] else output
                loss = torch.mean(criterion(output, label))

                # measure accuracy and mAP
                total_loss.append(loss.item())
                mAP.update(output.detach().cpu(), label.detach().cpu().int())
                data_loader.set_description(desc=f"Validate :: Loss:{loss.item():.04f}")

            map = mAP.compute()
            mean_total_loss = np.array(total_loss).mean()
            log.info(
                f"Validate :: Loss:{mean_total_loss:.4f}, mAP:{map:.4f}, Time:{data_loader.format_dict['elapsed']:.4f}"
            )
        return map
    
    def visualize(self, model, data_loader):
        model.eval()
        # print('CORRECT' if np.all((torch.round(F.softmax(output, dim=1)) == label).detach().cpu().numpy()) else 'NOT')

        for _, (image, label, metadata) in enumerate(data_loader):
            from util import GradCam
            grad_cam = GradCam(model=model, module='layer4', layer='2')
            mask = grad_cam(input[0].cuda(), None)
            import matplotlib.pyplot as plt
            import cv2
            
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.transpose(np.float32(input[0]).squeeze(), (2, 1, 0))
            cam = cam / np.max(cam)
            plt.imshow(np.uint8(255 * cam))
            plt.savefig('cam.png')
            plt.close()
            plt.imshow(np.uint8(heatmap * 255))
            plt.savefig('heatmap.png')

            cv2.imshow("cam", np.uint8(255 * cam))
            cv2.imshow("heatmap", np.uint8(heatmap * 255))
            cv2.waitKey()

    def update_lr(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.state["epoch"] != 0 and self.state["epoch"] in self.state["epoch_step"]:
            log.info('Update learning rate')
            for param_group in optimizer.state_dict()["param_groups"]:
                param_group["lr"] = param_group["lr"] * 0.1
                print(param_group["lr"])
