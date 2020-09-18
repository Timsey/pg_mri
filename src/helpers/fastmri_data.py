import pathlib
import random
import h5py
import numpy as np

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

        self.examples = []
        data_path = pathlib.Path(root)
        if 'brain' in str(data_path):
            self.dataset = 'brain'
        else:
            self.dataset = 'knee'
        # Using rss for Brain data
        self.recons_key = 'reconstruction_esc' if self.dataset == 'knee' \
            else 'reconstruction_rss'

        files = sorted(list(data_path.iterdir()))
        if sample_rate < 1:
            # if state is not None:  # Ensure same data is loaded when initialising the dataset multiple times in script
                # random.setstate(state)
            random.seed(0)  # TODO: This only works because we don't use random again in the same script: reset state
            random.shuffle(files)  # Same behaviour across runs is already guaranteed by setting the seed for random
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            # If 'acquisition' is specified, only slices from volumes that have been gathered using the specified
            # acquisition technique ('CORPD_FBK' or 'CORPDFS_FBK').
            if self.dataset == 'knee':
                if acquisition in ('CORPD_FBK', 'CORPDFS_FBK'):
                    with h5py.File(fname) as target:
                        if acquisition != target.attrs['acquisition']:
                            continue
                elif acquisition is not None:
                    raise ValueError("'acquisition' should be 'CORPD_FBK', 'CORPDFS_FBK', "
                                     "or None; not: {}".format(acquisition))
                kspace = h5py.File(fname, 'r')['kspace']
                num_slices = kspace.shape[0]
            else:  # Brain data, use all acquisition types? Only have gt stored for this dataset..
                gt = h5py.File(fname, 'r')[self.recons_key]
                num_slices = gt.shape[0]
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
            if self.dataset == 'knee':
                kspace = data['kspace'][slice]
            else:
                # Pad brain data up to 384 (max size) for consistency in crop later.
                res = 384  # Maximum size in train val test.
                bg = np.zeros((res, res), dtype=np.float32)
                w_pad = res - target.shape[-1]
                w_pad_left = w_pad // 2 if w_pad % 2 == 0 else w_pad // 2 + 1
                w_pad_right = w_pad // 2
                h_pad = res - target.shape[-2]
                h_pad_top = h_pad // 2 if h_pad % 2 == 0 else h_pad // 2 + 1
                h_pad_bot = h_pad // 2

                bg[h_pad_top:res - h_pad_bot, w_pad_left:res - w_pad_right] = target

                target = bg
                kspace = None  # Could make this np.fft.rfft2(gt) to have original_setting work with brain data?

            return self.transform(kspace, target, data.attrs, fname.name, slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=False, original_setting=True, low_res=False):
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
        self.low_res = low_res  # Whether to use low res full image or high res small image when cropping
        self.original_setting = original_setting  # Use original fix_kspace + image crop setting

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

        if self.original_setting:
            # Uses kspace passed to this function as basis. Transforms to image space, then abs + crop, transform back
            # to obtain kspace used (self.fix_kspace) in rest of algorithm. Implementation is artefact of experiments
            # with uncertainty based methods.
            # Note: abs(crop(ifft2(kspace))) == target (errors of order 1/500 of minimum value in either image)
            kspace = transforms.to_tensor(kspace)
            kspace = self.fix_kspace(kspace)  # We need this for Active Learning with uncertainty methods
            target = transforms.to_tensor(target)
            # In case resolution is not 320
            target = transforms.center_crop(target, (self.resolution, self.resolution))
        else:
            # Now obtain kspace from target for consistency between knee and brain datasets.
            # Target is used as ground truth as before.
            target = transforms.to_tensor(target)
            # TODO: Brain data has various resolutions, ranging from 213 to 384. Add filter to SliceData?
            if self.low_res:
                # Downscale image in kspace, to obtain low res full image, rather than high res small image
                kspace = transforms.rfft2(target)
                # Crop in kspace to obtain low resolution image of full body part
                kspace = transforms.complex_center_crop(kspace, (self.resolution, self.resolution))
                target = transforms.complex_abs(transforms.ifft2(kspace))
            else:  # Crop in image space
                # Note: this is very similar to original_setting: pretty much equivalent.
                # Thus, setting original_setting True, low_res False for Knee is same as setting original_setting False
                # low_rest False for Brain!
                #  For Brain we use the given fastMRI target as target. We then construct kspace from this by taking an
                #  rfft.
                #  For Knee we use the given fastMRI target as target. We also obtain kspace as an rfft from the target,
                #  but here we slightly differ from the above: the target we obtain the kspace from, is one constructed
                #  from the original kspace (by fix_kspace: ifft2, crop, abs, (rff2)). This is technically different,
                #  but since the fastMRI singlecoil kspace is obtained as an fft2 from the fastMRI target, we are
                #  essentially doing the same as directly taking an rfft on the target (experimentally the different
                #  computation gives a maximum difference a factor 1/500 smaller than the target value: 2-3 orders of
                #  magnitude, which should be negligible.).
                target = transforms.center_crop(target, (self.resolution, self.resolution))
                kspace = transforms.rfft2(target)

        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        zf = transforms.ifft2(masked_kspace)
        # Take real or complex abs to get a real image (this should be almost exactly the real part of the above)
        # TODO: Take real part or complex abs here?
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
        transform=DataTransform(train_mask, args.resolution, args.challenge, use_seed=False,
                                original_setting=args.original_setting, low_res=args.low_res),
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
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True,
                                original_setting=args.original_setting, low_res=args.low_res),
        sample_rate=dev_sample_rate,
        challenge=args.challenge,
        acquisition=args.acquisition,
        center_volume=args.center_volume,
        state=args.dev_state
    )
    test_data = SliceData(
        root=test_path,
        transform=DataTransform(test_mask, args.resolution, args.challenge, use_seed=True,
                                original_setting=args.original_setting, low_res=args.low_res),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        acquisition=args.acquisition,
        center_volume=args.center_volume,
        state=args.test_state
    )
    return train_data, dev_data, test_data
