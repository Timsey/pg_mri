import torch

from .unet_dist_model import UnetModelParam
from src.helpers import transforms

# import sys
# sys.path.append("home/timsey/Projects/bayesmri/train_bayes_unet_param")
# from train_bayes_unet_param import Arguments


def build_recon_model(recon_args, args):
    gauss_model = UnetModelParam(
        in_chans=1,
        out_chans=1,
        chans=recon_args.num_chans,
        num_pool_layers=recon_args.num_pools,
        drop_prob=recon_args.drop_prob
    ).to(args.device)

    # No gradients for this model
    for param in gauss_model.parameters():
        param.requires_grad = False
    return gauss_model


class Arguments:
    """
    Required to load the reconstruction model. Pickle requires the class definition to be visible/importable
    when loading a checkpoint containing an instance of that class.
    """
    def __init__(self):
        pass


def load_recon_model(args):
    checkpoint = torch.load(args.recon_model_checkpoint)
    recon_args = checkpoint['args']
    recon_model = build_recon_model(recon_args, args)
    if args.data_parallel:
        recon_model = torch.nn.DataParallel(recon_model)
    recon_model.load_state_dict(checkpoint['model'])
    del checkpoint
    return recon_args, recon_model


def normalize_instance_batch(data, eps=0.):
    # Normalises instances over last two dimensions (other dimensions are assumed to be batch dimensions)
    mean = data.mean(dim=(-2, -1), keepdim=True)
    std = data.std(dim=(-2, -1), keepdim=True)
    return transforms.normalize(data, mean, std, eps), mean, std


def get_new_zf(masked_kspace_batch):
    # Inverse Fourier Transform to get zero filled solution
    image_batch = transforms.ifft2(masked_kspace_batch)
    # Absolute value
    image_batch = transforms.complex_abs(image_batch)
    # Normalize input
    image_batch, means, stds = normalize_instance_batch(image_batch, eps=1e-11)
    image_batch = image_batch.clamp(-6, 6)
    return image_batch, means, stds


def acquire_new_zf(full_kspace, masked_kspace, next_row):
    # Acquire row
    cloned_masked_kspace = masked_kspace.clone()
    # Acquire row for all samples in the batch
    # shape = (batch_dim, column, row, complex)
    cloned_masked_kspace[..., next_row, :] = full_kspace[..., next_row, :]
    zero_filled, mean, std = get_new_zf(cloned_masked_kspace)
    return zero_filled, mean, std


def acquire_new_zf_exp(kspace, masked_kspace_exp):
    # Acquire row
    indices = list(range(masked_kspace_exp.size(0)))
    for index in indices:
        masked_kspace_exp[index, :, index, :] = kspace[0, :, index, :]

    zero_filled_exp, mean_exp, std_exp = get_new_zf(masked_kspace_exp)
    return zero_filled_exp, mean_exp, std_exp