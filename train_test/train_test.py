import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import os
import torchvision
import torchvision.transforms as transforms
import model.CNN as CNN
import data_loading.dataloading as dataloading
import tuning

device = 'cuda' if torch.cuda.is_available() else 'cpu'

