from typing import Tuple, Union
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


def get_PIL_image(image_path: Path) -> Image:
    try:
        return Image.open(image_path)
    except OSError as e:
        print(f"Error opening image: {image_path}")
        return
    


class LabelledDataset(Dataset):
    '''Basically implements torchvision.datasets.ImageFolder'''

    def __init__(self, folder_path: Path, transforms: transforms = None):
        super().__init__()

        self.fire_file_paths = [folder_path / 'wildfire' / name for name in os.listdir(folder_path / 'wildfire')]
        self.no_fire_file_paths = [folder_path / 'nowildfire' / name for name in os.listdir(folder_path / 'nowildfire')]
        self.transforms = transforms

    def __len__(self,) -> int:
        return len(self.fire_file_paths) + len(self.no_fire_file_paths)
    
    def __getitem__(self, index) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
        if index < len(self.no_fire_file_paths):
            image = get_PIL_image(self.no_fire_file_paths[index])
            label = 0
        else:
            index = index - len(self.no_fire_file_paths)
            image = get_PIL_image(self.fire_file_paths[index])
            label = 1

        if self.transforms and image is not None:
            try:
                return self.transforms(image), label
            except: 
                print(type(image))
                print(np.array(image).shape)
                if index < len(self.no_fire_file_paths):
                    name = self.no_fire_file_paths[index]
                else:
                    index = index - len(self.no_fire_file_paths)
                    name = self.fire_file_paths[index]
                print('ERROR: ', name)
        return image, label
    

class UnlabelledDataset(Dataset):
    '''Basically implements torchvision.datasets.ImageFolder'''

    def __init__(self, folder_path: Path, transforms: transforms = None):
        super().__init__()

        self.file_paths = [folder_path / name for name in os.listdir(folder_path)]
        self.transforms = transforms

    def __len__(self,) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
        image = get_PIL_image(self.no_fire_file_paths[index])
        if self.transforms and image:
            return self.transforms(image)
        return image


def get_dataloader(
        folder_path: Path, 
        transforms: transforms = None, 
        dataset_type: str = 'labelled',
        batch_size: int = 32,
        shuffle: bool = True) -> DataLoader:

    if dataset_type == 'labelled':
        dataset = LabelledDataset(folder_path=folder_path, transforms=transforms)
    elif dataset_type == 'unlabelled':
        dataset = UnlabelledDataset(folder_path=folder_path, transforms=transforms)
    else:
        raise NotImplementedError('This dataset type is not implemented. Check string correctness')
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def get_train_transforms(config: SimpleNamespace, dataset_stats: str = 'unlabelled') -> transforms:

    if dataset_stats == 'imagenet':
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    elif dataset_stats == 'unlabelled':
        means = (0.297, 0.348, 0.253)
        stds = (0.195, 0.163, 0.168)
    elif dataset_stats == 'train':
        means = (0.296, 0.347, 0.252)
        stds = (0.193, 0.162, 0.166)
    else:
        raise NotImplementedError(f'No such dataset_stat {dataset_stats}')
        

    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(p=config.p_hor_flip),  
                                    transforms.RandomVerticalFlip(p=config.p_ver_flip),    
                                    transforms.RandomRotation(degrees=config.rand_rot_degree),  
                                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)
                                ])
    
    return transform


def get_val_transforms(dataset_stats: str = 'unlabelled') -> transforms:

    if dataset_stats == 'imagenet':
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    elif dataset_stats == 'unlabelled':
        means = (0.297, 0.348, 0.253)
        stds = (0.195, 0.163, 0.168)
    elif dataset_stats == 'train':
        means = (0.296, 0.347, 0.252)
        stds = (0.193, 0.162, 0.166)
    else:
        raise NotImplementedError(f'No such dataset_stat {dataset_stats}')
        

    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)
                                    ])
    
    return transform

