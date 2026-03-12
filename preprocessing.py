## Author: Aalam Sultanji
## Last updated: 23-02-2026
## Email: aalam.sultanji@gmail.com

'''
Preprocessing and cleaning of the data. Contains further helper functions to process any potential oddidites in the data. Also need to convert all images to black and white, resizing. 
Might look into using RGB images as an extension. 
'''

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='dataset\Data', 
                               transform=transform)

print(dataset.classes)
print(dataset.class_to_idx)