import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import torch.utils.data
from scipy.stats import entropy
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from data.IS_dataset import ISImageDataset
from PIL import Image

class ISImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files = get_paths_from_images(root)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        item_image = self.transform(img)
        return item_image

    def __len__(self):
        return len(self.files)
