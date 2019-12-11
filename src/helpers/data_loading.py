import numpy as np
import torch
import pathlib
import random
import h5py

from torch.utils.data import Dataset, DataLoader

from src.helpers import transforms


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, max_slices=None,
                 acquisition=None):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            # If 'acquisition' is specified, only slices from volumes that have been gathered using the specified
            # acquisition technique ('CORPD_FBK' or 'CORPDFS_FBK').
            if acquisition in ('CORPD_FBK', 'CORPDFS_FBK'):
                with h5py.File(fname) as target:
                    if acquisition != target.attrs['acquisition']:
                        continue
            elif acquisition is not None:
                raise ValueError("'acquisition' should be 'CORPD_FBK', 'CORPDFS_FBK', "
                                 "or None; not: {}".format(acquisition))
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]
            if max_slices is not None:
                if len(self.examples) > max_slices:
                    self.examples = self.examples[:max_slices]
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)


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


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=False):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
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

        Changed from original by mapping kspace onto 320x320 size and normalising based on the real valued images,
         before applying mask, for AL consistency.
        """
        kspace = transforms.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        kspace = self.fix_kspace(kspace)  # We need this for Active Learning
        masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        zf = transforms.ifft2(masked_kspace)
        # Take real or complex abs to get a real image (this should be almost exactly the real part of the above)
        # TODO: Take real part or complex abs here?
        zf = transforms.complex_abs(zf)
        # Normalize input
        zf, mean, std = transforms.normalize_instance(zf, eps=1e-11)
        zf = zf.clamp(-6, 6)

        target = transforms.to_tensor(target)
        # In case resolution is not 320
        target = transforms.center_crop(target, (self.resolution, self.resolution))
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)

        # Need to return kspace and mask information when doing active learning, since we are
        # acquiring frequencies and updating the mask for a data point during an AL loop.
        return kspace, masked_kspace, mask, zf, target, mean, std, attrs['norm'].astype(np.float32)

    def fix_kspace(self, kspace):
        """
        Since kspace does not have a constant size over all volumes, fastMRI center crops zero-filled images. However,
        this is done only after applying the mask to the full kspace, which leads to inconsistencies during active
        learning: in particular, the AL acquisition step involves predicting values of a kspace row, and adding that
        to kspace before reconstructing. Since model output has a size of self.resolution, the kspace row will have
        that size as well, and as such we require that we obtain a row of that size from the full kspace when
        doing our active learning loop. We cannot simply cut off parts of kspace, since fourier transforms are not
        local (i.e. a pixel in kspace influences every pixel in image space and vice-versa).

        Thus we must resize kspace to the image size the model uses, by first transforming to real space, taking a
        center crop, and then transforming back. This function does that.
        """

        def rfft2(data):
            data = transforms.ifftshift(data, dim=(-2, -1))
            data = torch.rfft(data, 2, normalized=True, onesided=False)
            data = transforms.fftshift(data, dim=(-3, -2))
            return data

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(kspace)
        # Crop input image to get correctly sized kspace
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Take complex abs to get a real image
        image = transforms.complex_abs(image)
        # rfft this image to get the kspace that will be used in active learning
        kspace = rfft2(image)
        return kspace


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask

    Additionally returns the used acceleration and center fraction for evaluation purposes.
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return data * mask, mask


def create_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    # test_mask = MaskFunc(args.center_fractions, args.accelerations)

    # We cannot use the original dataset when active learning, since we have no test data then.
    train_path = args.data_path / f'{args.challenge}_train_al'

    train_data = SliceData(
        root=train_path,
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        max_slices=args.max_train_slices,
        challenge=args.challenge,
        acquisition=args.acquisition
    )
    # use_seed=True ensures the same mask is used for all slices in a given volume every time. This means the
    # development set stays the same with every use. Note that this also means any metrics based on the mask will
    # be consistently evaluated on the same volumes every time.
    dev_data = SliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        max_slices=args.max_dev_slices,
        challenge=args.challenge,
        acquisition=args.acquisition
    )
    # test_data = SliceData(
    #     root=args.data_path / f'{args.challenge}_test_al',
    #     transform=DataTransform(test_mask, args.resolution, args.challenge, use_seed=True),
    #     sample_rate=args.sample_rate,
    #     max_slices=args.max_test_slices,
    #     challenge=args.challenge,
    #     acquisition=args.acquisition
    # )
    return dev_data, train_data, None  # test_data


def create_data_loaders(args):
    dev_data, train_data, test_data = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
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
    return train_loader, dev_loader, test_loader, display_loader
