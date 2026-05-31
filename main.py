## Author: Aalam Sultanji
## Last updated: 29-05-2026
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
from model.CNN import CNN1
from train_test.train_test import train, evaluate, test
from data_loading.dataloading import dataloaders
import numpy as np

'''
Main block, runs the training and testing of the model, includes the configurations. 
'''

#Configurations:
DATADIR = 'dataset/Data'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 5
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_weights, class_names = dataloaders(DATADIR, batch_size=BATCH_SIZE)
    model = CNN1(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #training the model
    model, history = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS)

    #testing the model
    model.load_state_dict(torch.load('weights/best_model.pth', weights_only=True))
    results = test(model, test_loader, device, class_names)




