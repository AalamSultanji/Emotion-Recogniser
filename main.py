## Author: Aalam Sultanji
## Last updated: 23-02-2026
## Email: aalam.sultanji@gmail.com

'''
Main file, runs the trainig, testing and evaluation of the model. 
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import os
import torchvision
import torchvision.transforms as transforms
import CNN
import preprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'


