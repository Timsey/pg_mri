import torch
from src.helpers import transforms

from src.recon_models.unet_dist_model import build_dist_model
from src.recon_models.unet_kengal_model import build_kengal_model


def recon_model_forward_pass(args, recon_model, zf):
    model_name = recon_model.__class__.__name__
    if args.recon_model_name == 'kengal_laplace':
        output = recon_model(zf)
    elif args.recon_model_name == 'dist_gauss':
        loc, logscale = recon_model(zf)
        output = torch.cat((loc, logscale), dim=1)
    else:
        raise ValueError('Model type {} is not supported'.format(model_name))
    # Output of size batch x channel x resolution x resolution
    return output


def load_recon_model(args):
    checkpoint = torch.load(args.recon_model_checkpoint)
    recon_args = checkpoint['args']
    if args.recon_model_name == 'kengal_laplace':
        recon_model = build_kengal_model(recon_args, args)
    elif args.recon_model_name == 'dist_gauss':
        recon_model = build_dist_model(recon_args, args)
    else:
        raise ValueError('Model name {} is not a valid option.'.format(args.recon_model_name))

    if recon_args.data_parallel:  # if model was saved with data_parallel
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


def acquire_new_zf_exp(k, mk, to_acquire):
    # Expand masked kspace over channel dimension to prepare for adding all kspace rows to acquire
    mk_exp = mk.expand(len(to_acquire), -1, -1, -1).clone()  # TODO: .clone() necessary here? Yes?
    # Acquire row
    for index, row in enumerate(to_acquire):
        mk_exp[index, :, row.item(), :] = k[0, :, row.item(), :]
    # Obtain zero filled image from all len(to_acquire) new kspaces
    zero_filled_exp, mean_exp, std_exp = get_new_zf(mk_exp)
    return zero_filled_exp, mean_exp, std_exp
