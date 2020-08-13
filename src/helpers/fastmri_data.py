import pathlib
import random
import h5py

from torch.utils.data import Dataset

from src.helpers import transforms


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, acquisition=None, center_volume=False, state=None):
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
        files = sorted(list(pathlib.Path(root).iterdir()))
        if sample_rate < 1:
            # if state is not None:  # Ensure same data is loaded when initialising the dataset multiple times in script
                # random.setstate(state)
            random.seed(0)
            random.shuffle(files)  # Same behaviour across runs is already guaranteed by setting the seed for random
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
            if center_volume:  # Only use the slices in the center half of the volume
                self.examples += [(fname, slice) for slice in range(num_slices // 4, 3 * num_slices // 4)]
            else:
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=False, real=True, low_res_320=True):
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
        self.real = real  # whether to use real valued k-space
        self.low_res_320 = low_res_320

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

        Changed from original: now starting from GT RSS, which makes more sense if doing singlecoil.
        """

        # Now obtain kspace from gt for consistency between knee and brain datasets
        target = transforms.to_tensor(target)
        assert target.size(-2) == target.size(-1) == 320  # Check data
        if self.low_res_320:
            kspace = transforms.fft2(target)
            # Crop in kspace to obtain low resolution image of full body part
            kspace = transforms.complex_center_crop(kspace, (self.resolution, self.resolution))
            target = transforms.complex_abs(transforms.ifft2(kspace))
        else:  # Crop in image space
            target = transforms.center_crop(target, (self.resolution, self.resolution))
            kspace = transforms.fft2(target)

        # Note: abs(crop(ifft2(kspace))) == target (errors of order 1/500 of minimum value in either image)
        # kspace = transforms.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        # kspace = self.fix_kspace(kspace)  # We need this for Active Learning
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        zf = transforms.ifft2(masked_kspace)
        # Take real or complex abs to get a real image (this should be almost exactly the real part of the above)
        # TODO: Take real part or complex abs here?
        zf = transforms.complex_abs(zf)
        # Normalize input
        zf, zf_mean, zf_std = transforms.normalize_instance(zf, eps=1e-11)
        zf = zf.clamp(-6, 6)

        target = transforms.to_tensor(target)
        # In case resolution is not 320
        target = transforms.center_crop(target, (self.resolution, self.resolution))
        # # Normalize target
        # target = transforms.normalize(target, mean, std, eps=1e-11)
        target, gt_mean, gt_std = transforms.normalize_instance(target, eps=1e-11)
        target = target.clamp(-6, 6)

        # Need to return kspace and mask information when doing active learning, since we are
        # acquiring frequencies and updating the mask for a data point during an AL loop.
        return kspace, masked_kspace, mask, zf, target, gt_mean, gt_std, fname, slice  # , zf_mean, zf_std

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

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(kspace)
        # Crop input image to get correctly sized kspace
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # kspace = transforms.fft2(image)
        # Take complex abs to get a real image
        image = transforms.complex_abs(image)
        # rfft this image to get the kspace that will be used in active learning
        rkspace = transforms.rfft2(image)
        # kspace = transforms.fft2(image)
        return rkspace


def create_fastmri_datasets(args, train_mask, dev_mask, test_mask):
    train_path = args.data_path / f'{args.challenge}_train_al'
    dev_path = args.data_path / f'{args.challenge}_val'
    test_path = args.data_path / f'{args.challenge}_test_al'

    train_data = SliceData(
        root=train_path,
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        acquisition=args.acquisition,
        center_volume=args.center_volume,
        state=args.train_state
    )
    # use_seed=True ensures the same mask is used for all slices in a given volume every time. This means the
    # development set stays the same with every use. Note that this also means any metrics based on the mask will
    # be consistently evaluated on the same volumes every time.
    mult = 2 if args.sample_rate == 0.04 else 1  # TODO: this is now hardcoded to get more validation samples: fix this
    dev_sample_rate = args.sample_rate * mult
    dev_data = SliceData(
        root=dev_path,
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=dev_sample_rate,
        challenge=args.challenge,
        acquisition=args.acquisition,
        center_volume=args.center_volume,
        state=args.dev_state
    )
    test_data = SliceData(
        root=test_path,
        transform=DataTransform(test_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        acquisition=args.acquisition,
        center_volume=args.center_volume,
        state=args.test_state
    )
    return train_data, dev_data, test_data
