import torch
import pathlib
import random
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.helpers import transforms


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, dataset, sample_rate=1, acquisition=None, center_volume=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
        """
        self.transform = transform

        self.examples = []

        self.dataset = dataset
        assert dataset in ['knee', 'brain'], f"Dataset must be 'knee'' or 'brain'', not {dataset}"
        # Using rss for Brain data
        self.recons_key = 'reconstruction_esc' if self.dataset == 'knee' \
            else 'reconstruction_rss'

        data_path = pathlib.Path(root)
        files = sorted(list(data_path.iterdir()))
        if sample_rate < 1:
            # Make sure to always use the same dataset, even when the random seed is different.
            state = random.getstate()
            random.seed(0)
            random.shuffle(files)
            # Reset random state to make sure other calls of random in the same run still use the correct seed
            random.setstate(state)
            # Sample data volumes
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            # If 'acquisition' is specified, only slices from volumes that have been gathered using the specified
            # acquisition technique ('CORPD_FBK' or 'CORPDFS_FBK').
            # Brain data uses all acquisition types.
            if self.dataset == 'knee':
                if acquisition in ('CORPD_FBK', 'CORPDFS_FBK'):
                    with h5py.File(fname) as target:
                        if acquisition != target.attrs['acquisition']:
                            continue
                else:
                    assert acquisition is None, ("'acquisition' should be 'CORPD_FBK', 'CORPDFS_FBK', "
                                                 "or None; not: {}".format(acquisition))

            target = h5py.File(fname, 'r')[self.recons_key]
            num_slices = target.shape[0]

            if center_volume:  # Only use the slices in the center half of the volume
                self.examples += [(fname, slice) for slice in range(num_slices // 4, 3 * num_slices // 4)]
            else:
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            target = data[self.recons_key][slice] if self.recons_key in data else None
            if self.dataset == 'brain':  # TODO: for knee data as well?
                # Pad brain data up to 384 (max size) for consistency in crop later.
                res = 384  # Maximum size of brain data slices
                bg = np.zeros((res, res), dtype=np.float32)
                w_pad = res - target.shape[-1]
                w_pad_left = w_pad // 2 if w_pad % 2 == 0 else w_pad // 2 + 1
                w_pad_right = w_pad // 2
                h_pad = res - target.shape[-2]
                h_pad_top = h_pad // 2 if h_pad % 2 == 0 else h_pad // 2 + 1
                h_pad_bot = h_pad // 2
                bg[h_pad_top:res - h_pad_bot, w_pad_left:res - w_pad_right] = target
                target = bg

            return self.transform(target, data.attrs, fname.name, slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, use_seed=False):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, target, attrs, fname, slice):
        """
        Args:
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.

        Additionally returns the used acceleration and center fraction for evaluation purposes.

        Changed from original: now starting from GT RSS, which makes more sense if doing singlecoil.
        """

        # Obtain full kspace from ground truth
        target = transforms.to_tensor(target)
        target = transforms.center_crop(target, (self.resolution, self.resolution))
        kspace = transforms.rfft2(target)

        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        zf = transforms.ifft2(masked_kspace)
        # Take complex abs to get a real image
        zf = transforms.complex_abs(zf)
        # Normalize input
        zf, zf_mean, zf_std = transforms.normalize_instance(zf, eps=1e-11)
        zf = zf.clamp(-6, 6)

        # # Normalize target
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        target, gt_mean, gt_std = transforms.normalize_instance(target, eps=1e-11)
        target = target.clamp(-6, 6)

        # Need to return kspace and mask information when doing active learning, since we are
        # acquiring frequencies and updating the mask for a data point during an AL loop.
        return kspace, masked_kspace, mask, zf, target, gt_mean, gt_std, fname, slice


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


def create_fastmri_dataset(args, partition):
    # TODO: Paths not hardcoded?
    if partition == 'train':
        path = args.data_path / f'singlecoil_train_al'
        use_seed = False
    elif partition == 'val':
        path = args.data_path / f'singlecoil_val'
        use_seed = True
    elif partition == 'test':
        path = args.data_path / f'singlecoil_test_al'
        use_seed = True
    else:
        raise ValueError(f"partition should be in ['train', 'val', 'test'], not {partition}")

    mask = MaskFunc(args.center_fractions, args.accelerations)

    dataset = SliceData(
        root=path,
        transform=DataTransform(mask, args.resolution, use_seed=use_seed),
        dataset=args.dataset,
        sample_rate=args.sample_rate,
        acquisition=args.acquisition,
        center_volume=args.center_volume
    )

    return dataset


def create_data_loader(args, partition, shuffle=False):
    # TODO: set shuffle to True for train
    dataset = create_fastmri_dataset(args, partition)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader


# def create_fastmri_datasets(args, train_mask, dev_mask, test_mask):
#     # TODO: This not hardcoded?
#     train_path = args.data_path / f'singlecoil_train_al'
#     dev_path = args.data_path / f'singlecoil_val'
#     test_path = args.data_path / f'singlecoil_test_al'
#
#     train_data = SliceData(
#         root=train_path,
#         transform=DataTransform(train_mask, args.resolution, use_seed=False),
#         dataset=args.dataset,
#         sample_rate=args.sample_rate,
#         acquisition=args.acquisition,
#         center_volume=args.center_volume
#     )
#
#     dev_data = SliceData(
#         root=dev_path,
#         transform=DataTransform(dev_mask, args.resolution, use_seed=True),
#         dataset=args.dataset,
#         sample_rate=args.sample_rate,
#         acquisition=args.acquisition,
#         center_volume=args.center_volume
#     )
#
#     test_data = SliceData(
#         root=test_path,
#         transform=DataTransform(test_mask, args.resolution, use_seed=True),
#         dataset=args.dataset,
#         sample_rate=args.sample_rate,
#         acquisition=args.acquisition,
#         center_volume=args.center_volume,
#     )
#     return train_data, dev_data, test_data


# def create_datasets(args):
#     train_mask = MaskFunc(args.center_fractions, args.accelerations)
#     dev_mask = MaskFunc(args.center_fractions, args.accelerations)
#     test_mask = MaskFunc(args.center_fractions, args.accelerations)
#
#     train_data, dev_data, test_data = create_fastmri_datasets(args, train_mask, dev_mask, test_mask)
#
#     return train_data, dev_data, test_data


# def create_data_loaders(args, shuffle_train=True):
#     train_data, dev_data, test_data = create_datasets(args)
#     logging.info('Train slices: {}, Dev slices: {}, Test slices: {}'.format(
#         len(train_data), len(dev_data), len(test_data)))
#
#     train_loader = DataLoader(
#         dataset=train_data,
#         batch_size=args.batch_size,
#         shuffle=shuffle_train,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#     dev_loader = DataLoader(
#         dataset=dev_data,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#
#     test_loader = DataLoader(
#         dataset=test_data,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#     return train_loader, dev_loader, test_loader
