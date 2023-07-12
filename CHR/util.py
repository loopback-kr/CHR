import sys, os, json, pickle, csv, re, random, logging, importlib, argparse, math
from datetime import datetime
from os.path import join, basename, exists, splitext, dirname, isdir, isfile
import matplotlib.pyplot as plt


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