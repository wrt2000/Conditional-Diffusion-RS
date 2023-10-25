from io import BytesIO
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import data.util as Util
import numpy as np
import time as Date
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from data.util import get_paths_from_images, transform_augment

IMG_MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)   


class RSDataset(Dataset):
    def __init__(
        self,
        dataset_opt,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.dataset_opt = dataset_opt
        self.dataroot = dataset_opt["dataroot"]
        self.transform = transform
        self.datatype = dataset_opt["datatype"]

        self.img_path = get_paths_from_images('{}/images'.format(self.dataroot))
        self.mask_path = get_paths_from_images('{}/masks'.format(self.dataroot))

    def __len__(self) -> int:
        return len(self.img_path)
    

    def __getitem__(self, index) -> Any:
        begin = Date.time()
        
        image = Image.open(self.img_path[index]).convert('RGB')
        mask = Image.open(self.mask_path[index]).convert('RGB')
        assert image.size == mask.size, "image size: {}, mask size: {}".format(image.size, mask.size)

        [image, mask] = transform_augment([image, mask], self.datatype, min_max=(-1, 1))


        return {'Image': image, 'Mask': mask, 'Index': index}

def create_dataloader(dataset, phase):
    '''create dataloader '''
    dataset_opt = dataset.dataset_opt
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return DataLoader(
            dataset, batch_size=dataset_opt["batch_size"], shuffle=False, num_workers=dataset_opt["num_workers"], pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))