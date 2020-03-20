import numpy as np
import torch
import logging
import random

from collections import namedtuple
from torch.utils.data import DataLoader

from src.helpers.fastmri_data import create_fastmri_datasets
from src.helpers.cifar_data import create_cifar10_datasets


# TrainPair = namedtuple('TrainPair', ('step', 'predictions', 'target'))

# class ReplayMemory:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, step, predictions, target):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)  # Add empty entry to overwrite if memory not full yet
#         self.memory[self.position] = TrainPair(step, predictions, target)
#         self.position = (self.position + 1) % self.capacity
# 
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)


class MaskFunc:
    """
    MaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.

        Additionally returns the used acceleration and center fraction for evaluation purposes.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def create_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    # test_mask = MaskFunc(args.center_fractions, args.accelerations)

    if args.dataset == 'fastmri':
        train_data, dev_data = create_fastmri_datasets(args, train_mask, dev_mask)
    elif args.dataset == 'cifar10':
        train_data, dev_data = create_cifar10_datasets(args, train_mask, dev_mask)
    else:
        raise ValueError("Invalid dataset {}".format(args.dataset))

    return train_data, dev_data


def create_data_loaders(args, shuffle_train=True):
    train_data, dev_data = create_datasets(args)
    logging.info('Train slices: {}, Dev slices: {}'.format(len(train_data), len(dev_data)))

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Use dev data for visualisation purposes.
    # Batch size is set to 16 to get a 4x4 grid of images.
    display_loader = DataLoader(
        dataset=dev_data,
        batch_size=16,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, dev_loader, None, display_loader
