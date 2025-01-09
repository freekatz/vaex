import glob
import os

import PIL.Image as PImage
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class UnlabeledDatasetFolder(DatasetFolder):
    def __init__(self, root, transform=None, split='train'):
        self.root = os.path.join(root, split)
        super().__init__(root=self.root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class FFHQ(Dataset):
    def __init__(self, root, transform=None, split='train'):
        super().__init__()
        self.root = root
        split_file = os.path.join(self.root, f'ffhq_{split}.txt')
        with open(split_file, 'r') as file:
            self.samples = [os.path.join(self.root, line.strip()) for line in file.readlines() if line.endswith('.png')]
        assert(len(self.samples) > 0)

        self.transform = transform
        self.loader = pil_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
