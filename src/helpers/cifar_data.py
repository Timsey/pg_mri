import numpy as np
import torch
import random

import torchvision
import torchvision.transforms as trfs
from torch.utils.data import Dataset

from src.helpers import transforms


class CifarData(Dataset):
    def __init__(self, cifar, train, transform, label='frog', sample_rate=1):
        self.transform = transform

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        class_ind = classes.index(label)
        if train:
            indices = np.where(np.array(cifar.train_labels) == class_ind)[0]
            self.images = cifar.train_data[indices, ...]
        else:
            indices = np.where(np.array(cifar.test_labels) == class_ind)[0]
            self.images = cifar.test_data[indices, ...]

        if sample_rate < 1:
            random.shuffle(self.images)
            num_files = round(len(self.images) * sample_rate)
            self.images = self.images[:num_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        return self.transform(image)


class CifarTransform:
    def __init__(self, mask_func, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(self, image):
        pil_image = trfs.functional.to_pil_image(image)
        pil_target = trfs.functional.to_grayscale(pil_image, num_output_channels=1)
        target = transforms.to_tensor(np.array(pil_target)).float() / 255
        kspace = self.rfft2(target)

        # Mask kspace
        seed = None if not self.use_seed else tuple(map(ord, str(np.mean(image))))
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)

        # Inverse Fourier Transform to get zero filled solution
        zf = transforms.ifft2(masked_kspace)
        zf = transforms.complex_abs(zf)
        zf, mean, std = transforms.normalize_instance(zf, eps=1e-11)
        zf = zf.clamp(-6, 6)

        # Normalise target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return kspace, masked_kspace, mask, zf, target, mean, std, 0

    def rfft2(self, data):
        data = transforms.ifftshift(data, dim=(-2, -1))
        data = torch.rfft(data, 2, normalized=True, onesided=False)
        data = transforms.fftshift(data, dim=(-3, -2))
        return data


def create_cifar10_datasets(args, train_mask, dev_mask):
    train_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    train_data = CifarData(
        cifar=train_cifar,
        train=True,
        transform=CifarTransform(train_mask),
        sample_rate=args.sample_rate)

    dev_data = CifarData(
        cifar=test_cifar,
        train=False,
        transform=CifarTransform(dev_mask),
        sample_rate=args.sample_rate)

    return train_data, dev_data
