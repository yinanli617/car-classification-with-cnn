import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import timm

import time
import os
import sys
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import PIL.Image as Image
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_DIR = './stanford-car-classification/car_data/car_data/'
NUM_CAR_CLASSES = 196
