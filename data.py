import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose, Resize, RandomCrop
import numpy as np
from os import listdir
from os.path import exists
from numpy import zeros_like
from glob import glob
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

class AdaptiveResize():
    def __init__(self, size):
        self.int_resize = Resize(size, InterpolationMode.NEAREST)
        self.float_resize = Resize(size, InterpolationMode.BILINEAR)

    def __call__(self, tensor):
        if tensor.dtype == torch.long:
            return self.int_resize(tensor)
        else:
            return self.float_resize(tensor)

class StatefulRandomCrop():
    def __init__(self, size):
        self.random_crop = RandomCrop(size)
        self.state = None

    def __call__(self, tensor):
        if self.state == None:
            self.state = torch.random.get_rng_state()
        else:
            torch.random.set_rng_state(self.state)
            self.state = None
        return self.random_crop(tensor)

def none_transform(x):
    return x

def prepare_cross_validation_datasets(root_dir, cross_validation_index=0, common_transform=none_transform, img_transform=none_transform, seg_transform=none_transform):
    
    surgery_types = listdir(f"{root_dir}/Training")
    surgery_indices_per_type = [sorted(map(int, listdir(f"{root_dir}/Training/{surgery_type}"))) for surgery_type in surgery_types]

    training_surgeries = []
    validation_surgeries = []

    for surgery_type, indicies in zip(surgery_types, surgery_indices_per_type):
        validation_surgeries += [f"{root_dir}/Training/{surgery_type}/{indicies.pop(cross_validation_index)}"]
        training_surgeries += [f"{root_dir}/Training/{surgery_type}/{index}" for index in indicies]

    training_dataset = ConcatDataset([RobustDataset(f"{surgery_path}/*", common_transform, img_transform, seg_transform) for surgery_path in training_surgeries])
    validation_dataset = ConcatDataset([RobustDataset(f"{surgery_path}/*", common_transform, img_transform, seg_transform) for surgery_path in validation_surgeries])

    return training_dataset, validation_dataset

class RobustDataset(Dataset):

    def __init__(self, sample_path_glob, common_transform=none_transform, img_transform=none_transform, seg_transform=none_transform):

        self.sample_paths = glob(sample_path_glob)

        self.img_transform = Compose([common_transform, img_transform])
        self.seg_transform = Compose([common_transform, seg_transform])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):

        sample_path = self.sample_paths[index]

        img_path = f"{sample_path}/raw.png"
        seg_path = f"{sample_path}/instrument_instances.png"

        img = np.array(Image.open(img_path))
        seg = np.array(Image.open(seg_path)) if exists(seg_path) else zeros_like(img[..., 0])

        img = torch.from_numpy(img).permute(2, 0, 1) / 255
        seg = torch.from_numpy(seg).unsqueeze(0).long()

        img = self.img_transform(img)
        seg = self.seg_transform(seg)
        
        seg[seg != 0] = 1
        seg = seg.squeeze()
        seg = one_hot(seg.squeeze(), num_classes=2).permute(2, 0, 1).float()

        return img, seg
