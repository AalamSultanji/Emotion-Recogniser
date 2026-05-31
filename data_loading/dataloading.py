## Author: Aalam Sultanji
## Last updated: 29-05-2026
## Email: aalam.sultanji@gmail.com

'''
Preprocessing and cleaning of the data. Contains further helper functions to process any potential oddidites in the data. Also need to convert all images to black and white, resizing. 
Might look into using RGB images as an extension. 
'''

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def dataloaders(dataset_dir, batch_size=32, train_size=0.7, val_size=0.15):
    '''
    This function takes in the dataset and transforms it for all three splits and returns the dataloaders. 

    '''
    #initial transformation: resizing and greyscaling. 
    train_transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

    train_set = datasets.ImageFolder(root=dataset_dir, 
                                transform=train_transform)
    test_set = datasets.ImageFolder(root=dataset_dir, 
                                transform=test_transform)
    class_names = train_set.classes
    #splitting the sets
    total = len(train_set)
    indices = list(range(total))
    np.random.shuffle(indices)
    train_len = int(train_size * total)
    val_len = int(val_size * total)

    train_index = indices[:train_len]
    val_index = indices[train_len:val_len+train_len]
    test_index = indices[val_len+train_len:]

    #creating the splits
    train_split = Subset(train_set, train_index)
    val_split = Subset(test_set, val_index)
    test_split = Subset(test_set, test_index)

    #creating the dataloaders
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    #computing the class weights for imbalanced data
    class_counts = np.bincount(np.array(train_set.targets)[train_index])
    class_weights = []
    for classes in range(len(class_counts)):
        class_weight = (class_counts.sum()) / (len(class_counts) * class_counts[classes])
        class_weights.append(class_weight)
    class_weights = torch.FloatTensor(class_weights)
    return train_loader, val_loader, test_loader, class_weights, class_names






