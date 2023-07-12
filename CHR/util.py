import sys, os, json, pickle, csv, re, random, logging, importlib, argparse, math
from datetime import datetime
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image


def create_logger():
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger()
    logger.propagate=False
    logger.setLevel(logging.DEBUG)
    stream_heandler = logging.StreamHandler()
    stream_heandler.setLevel(logging.INFO)
    stream_heandler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(stream_heandler)
    file_handler = logging.FileHandler(filename=join('logs', f'{datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s - %(module)s %(funcName)s (%(filename)s:%(lineno)d) @ %(processName)s=%(process)d'))
    logger.addHandler(file_handler)
    logging.getLogger('matplotlib.font_manager').disabled = True
    return logger
log = create_logger()

def visuzlize_metrics(args, metrics):
    plt.figure()
    plt.xlabel('Epoch')
    plt.xlim(1, args.epochs+1)
    plt.ylabel('Loss')
    plt.plot(metrics['loss_train'], label='Train loss', color='b')
    plt.plot(metrics['loss_valid'], label='Valid loss', color='g')
    plt.legend()
    plt.savefig(join('logs', f'metrics-{args.log_name}.png'))
    plt.close()

class Warp: # TODO: replace to resize transforms
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return (
            self.__class__.__name__
            + " (size={size}, interpolation={interpolation})".format(
                size=self.size, interpolation=self.interpolation
            )
        )
