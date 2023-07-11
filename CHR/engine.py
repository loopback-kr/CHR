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
import torchnet as tnt
from torch.utils.data import DataLoader
import torch.optim
import numpy as np
from tqdm import tqdm
from util import APMeter2


class Engine:
    def __init__(self, state={}):
        self.state = state
        # meters
        self.state["meter_loss"] = tnt.meter.AverageValueMeter()
        # time measure
        self.state["batch_time"] = tnt.meter.AverageValueMeter()
        self.state["data_time"] = tnt.meter.AverageValueMeter()

        self.state["ap_meter"] = APMeter2(difficult_examples=True)

    def reset_logger(self):
        self.state["meter_loss"].reset()
        self.state["batch_time"].reset()
        self.state["data_time"].reset()
        self.state["ap_meter"].reset()

    def print_metrics(self, is_train, verbose=False):
        map = 100 * self.state["ap_meter"].value().mean()
        loss = self.state["meter_loss"].value()[0]

        if verbose:
            if is_train:
                # print(model.module.spatial_pooling)
                # print(self.state['epoch'], loss.cpu().numpy()[0], map)
                print(
                    "Epoch: [{0}]\t"
                    "Loss {loss:.4f}\t"
                    "mAP {map:.3f}".format(
                        self.state["epoch"], loss=loss.cpu().numpy(), map=map
                    )
                )
            else:
                # print(self.state['ap_meter'].value())
                print(
                    "Test: \t Loss {loss:.4f}\t  mAP {map:.3f}".format(
                        loss=loss.cpu().numpy(), map=map
                    )
                )
        return map

    def on_start_batch(self):
        self.state["target_gt"] = self.state["target"].clone()
        self.state["target"][self.state["target"] == 0] = 1
        self.state["target"][self.state["target"] == -1] = 0

        input = self.state["input"]
        self.state["input"] = input[0]
        self.state["name"] = input[1]


    def on_end_batch(self, is_train, data_loader, verbose=False):
        # record loss
        self.state["loss_batch"] = self.state["loss"].data
        self.state["meter_loss"].add(self.state["loss_batch"].detach().cpu())

        if (verbose
            # and self.state["print_freq"] != 0
            # and self.state["iteration"] % self.state["print_freq"] == 0
        ):
            loss = self.state["meter_loss"].value()[0]
            batch_time = self.state["batch_time"].value()[0]
            data_time = self.state["data_time"].value()[0]
            if is_train:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time_current:.3f} ({batch_time:.3f})\t"
                    "Data {data_time_current:.3f} ({data_time:.3f})\t"
                    "Loss {loss_current:.4f} ({loss:.4f})".format(
                        self.state["epoch"],
                        self.state["iteration"],
                        len(data_loader),
                        batch_time_current=self.state["batch_time_current"],
                        batch_time=batch_time,
                        data_time_current=self.state["data_time_batch"],
                        data_time=data_time,
                        loss_current=self.state["loss_batch"],
                        loss=loss,
                    )
                )
            else:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time_current:.3f} ({batch_time:.3f})\t"
                    "Data {data_time_current:.3f} ({data_time:.3f})\t"
                    "Loss {loss_current:.4f} ({loss:.4f})".format(
                        self.state["iteration"],
                        len(data_loader),
                        batch_time_current=self.state["batch_time_current"],
                        batch_time=batch_time,
                        data_time_current=self.state["data_time_batch"],
                        data_time=data_time,
                        loss_current=self.state["loss_batch"],
                        loss=loss,
                    )
                )
        # measure mAP
        self.state["ap_meter"].add(
            self.state["output"][0].data, self.state["target_gt"]
        )

        if (verbose
            # and self.state["print_freq"] != 0
            # and self.state["iteration"] % self.state["print_freq"] == 0
        ):
            loss = self.state["meter_loss"].value()[0]
            batch_time = self.state["batch_time"].value()[0]
            data_time = self.state["data_time"].value()[0]
            if is_train:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time_current:.3f} ({batch_time:.3f})\t"
                    "Data {data_time_current:.3f} ({data_time:.3f})\t"
                    "Loss {loss_current:.4f} ({loss:.4f})".format(
                        self.state["epoch"],
                        self.state["iteration"],
                        len(data_loader),
                        batch_time_current=self.state["batch_time_current"],
                        batch_time=batch_time,
                        data_time_current=self.state["data_time_batch"],
                        data_time=data_time,
                        loss_current=self.state["loss_batch"],
                        loss=loss,
                    )
                )
            else:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time_current:.3f} ({batch_time:.3f})\t"
                    "Data {data_time_current:.3f} ({data_time:.3f})\t"
                    "Loss {loss_current:.4f} ({loss:.4f})".format(
                        self.state["iteration"],
                        len(data_loader),
                        batch_time_current=self.state["batch_time_current"],
                        batch_time=batch_time,
                        data_time_current=self.state["data_time_batch"],
                        data_time=data_time,
                        loss_current=self.state["loss_batch"],
                        loss=loss,
                    )
                )

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = torch.autograd.Variable(self.state["input"]).cuda()
        target_var = torch.autograd.Variable(self.state["target"]).cuda()

        # compute output
        self.state["output"] = model(input_var)

        if training:
            self.state["loss"] = 0
            for i in range(3): # supervision 개수
                output_1 = self.state["output"][i].data
                n_class = output_1.size(1)
                n_batch = output_1.size(0)
                n_data = np.ones((n_batch, n_class))
                n_target = 1 - target_var.data.cpu().numpy()
                n_output = output_1.cpu().numpy()
                index = np.where(n_output < -20)
                if len(index[0]) == 0:
                    self.state["loss"] = self.state["loss"] + torch.mean(
                        criterion(self.state["output"][i], target_var)
                    )
                    continue
                n_data[index] = 0
                n_data = 1 - n_data
                n_data = (
                    torch.autograd.Variable(torch.from_numpy(n_data)).float().cuda()
                )
                n_target = torch.autograd.Variable(torch.from_numpy(n_target)).cuda()
                mask = 1 - torch.mul(n_data, n_target)
                self.state["loss"] = self.state["loss"] + torch.mean(
                    torch.mul(mask, criterion(self.state["output"][i], target_var))
                )
            optimizer.zero_grad()
            self.state["loss"].backward()
            optimizer.step()
        else:
            self.state["loss"] = torch.mean(criterion(self.state["output"][0], target_var))
            return self.state["output"][0], target_var

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer):
        self.state["best_score"] = 0

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

        # optionally resume from a checkpoint
        if self.state["resume_model_path"] is not None:
            if os.path.isfile(self.state["resume_model_path"]):
                print("=> loading checkpoint '{}'".format(self.state["resume_model_path"]))
                checkpoint = torch.load(self.state["resume_model_path"])
                self.state["start_epoch"] = checkpoint["epoch"]
                self.state["best_score"] = checkpoint["best_score"]
                model.load_state_dict(checkpoint["state_dict"])
                print(
                    "=> loaded checkpoint (epoch {})".format(checkpoint["epoch"])
                )
            else:
                print("=> no checkpoint found at '{}'".format(self.state["resume_model_path"]))

        model = model.to(self.state["device"])
        criterion = criterion.to(self.state["device"])

        for epoch in range(self.state["start_epoch"], self.state["max_epochs"]):
            self.state["epoch"] = epoch
            self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = self.validate(valid_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state["best_score"]
            self.state["best_score"] = max(prec1, self.state["best_score"])
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": "Resnet",
                    # 'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                    "state_dict": model.state_dict(),
                    "best_score": self.state["best_score"],
                },
                is_best,
                filename=f"checkpoint_{epoch:02d}ep.pth"
            )
            print(" *** best={best:.3f}".format(best=self.state["best_score"]))
    
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

        model = model.to(self.state["device"])
        criterion = criterion.to(self.state["device"])

        prec1 = self.validate(test_loader, model, criterion)
        print('Final score:', prec1)

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train() # switch to train mode

        self.reset_logger()

        data_loader = tqdm(data_loader, desc="Training", dynamic_ncols=True)

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state["iteration"] = i
            self.state["data_time_batch"] = time.time() - end
            self.state["data_time"].add(self.state["data_time_batch"])

            self.state["input"] = input
            self.state["target"] = target

            self.on_start_batch()

            self.state["target"] = self.state["target"].cuda(non_blocking=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state["batch_time_current"] = time.time() - end
            self.state["batch_time"].add(self.state["batch_time_current"])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, data_loader)
        self.print_metrics(True)

    def validate(self, data_loader, model, criterion):
        # switch to evaluate mode
        model.eval()

        # for i, (input, target) in enumerate(data_loader):
        #     from util import GradCam
        #     grad_cam = GradCam(model=model, module='layer4', layer='2')
        #     mask = grad_cam(input[0].cuda(), None)
        #     import matplotlib.pyplot as plt
        #     import cv2
            
        #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        #     heatmap = np.float32(heatmap) / 255
        #     cam = heatmap + np.transpose(np.float32(input[0]).squeeze(), (2, 1, 0))
        #     cam = cam / np.max(cam)
        #     plt.imshow(np.uint8(255 * cam))
        #     plt.savefig('cam.png')
        #     plt.close()
        #     plt.imshow(np.uint8(heatmap * 255))
        #     plt.savefig('heatmap.png')

            # cv2.imshow("cam", np.uint8(255 * cam))
            # cv2.imshow("heatmap", np.uint8(heatmap * 255))
            # cv2.waitKey()

        with torch.no_grad():
            self.reset_logger()

            data_loader = tqdm(data_loader, desc='Validating', dynamic_ncols=True)

            end = time.time()
            for i, (input, target) in enumerate(data_loader):
                # measure data loading time
                self.state["iteration"] = i
                self.state["data_time_batch"] = time.time() - end
                self.state["data_time"].add(self.state["data_time_batch"])

                self.state["input"] = input
                self.state["target"] = target

                self.on_start_batch()

                self.state["target"] = self.state["target"].cuda(non_blocking=True)

                pred, gt = self.on_forward(False, model, criterion, data_loader)

                # measure elapsed time
                self.state["batch_time_current"] = time.time() - end
                self.state["batch_time"].add(self.state["batch_time_current"])
                end = time.time()
                # measure accuracy
                self.on_end_batch(False, data_loader, verbose=False)
                
                # print('CORRECT' if np.all((torch.round(F.softmax(pred, dim=1)) == gt).detach().cpu().numpy()) else 'NOT')
                # TODO: support batch infe
            score = self.print_metrics(False)
            
            return score

    def save_checkpoint(self, state, is_best, filename="checkpoint.pth", best_filename="model_best.pth"):
        if self.state["model_save_dir"] is not None:
            os.makedirs(self.state["model_save_dir"], exist_ok=True)
        print("save model {filename}".format(filename=filename))
        torch.save(state, join(self.state["model_save_dir"], filename))
        if is_best:
            copyfile(join(self.state["model_save_dir"], filename), join(self.state["model_save_dir"], best_filename))
            copyfile(join(self.state["model_save_dir"], filename), join(self.state["model_save_dir"], f"model_best_{state['best_score']:.4f}.pth"))

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        if self.state["epoch"] != 0 and self.state["epoch"] in self.state["epoch_step"]:
            print("update learning rate")
            for param_group in optimizer.state_dict()["param_groups"]:
                param_group["lr"] = param_group["lr"] * 0.1
                print(param_group["lr"])
