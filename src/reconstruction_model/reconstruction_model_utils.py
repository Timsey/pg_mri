import torch
import h5py
import numpy as np

from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.reconstruction_model.reconstruction_model_def import build_reconstruction_model
from src.helpers.utils import build_optim
from src.helpers import transforms


def load_recon_model(args, optim=False):
    checkpoint = torch.load(args.recon_model_checkpoint)
    recon_args = checkpoint['args']
    recon_model = build_reconstruction_model(recon_args)

    if not optim:
        # No gradients for this model
        for param in recon_model.parameters():
            param.requires_grad = False

    if recon_args.data_parallel:  # if model was saved with data_parallel
        recon_model = torch.nn.DataParallel(recon_model)
    recon_model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_optim(args, recon_model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        return recon_model, recon_args, start_epoch, optimizer

    del checkpoint
    return recon_args, recon_model


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


def change_target_resolution(args, target):
    if args.dataset == 'brain':
        # Pad brain data up to 384 (max size) for consistency in crop later.
        # This is also done in SliceData class when loading images for training and validation.
        # This is the batched version though.
        res = 384  # Maximum size in train val test.
        bg = np.zeros((target.shape[0], res, res), dtype=np.float32)
        w_pad = res - target.shape[-1]
        w_pad_left = w_pad // 2 if w_pad % 2 == 0 else w_pad // 2 + 1
        w_pad_right = w_pad // 2
        h_pad = res - target.shape[-2]
        h_pad_top = h_pad // 2 if h_pad % 2 == 0 else h_pad // 2 + 1
        h_pad_bot = h_pad // 2

        bg[:, h_pad_top:res - h_pad_bot, w_pad_left:res - w_pad_right] = target
        target = bg

    # Now obtain kspace from target for consistency between knee and brain datasets.
    # Target is used as ground truth as before.
    target = transforms.to_tensor(target)
    target = transforms.center_crop(target, (args.resolution, args.resolution))
    return target.numpy()


# def acquire_new_zf(full_kspace, masked_kspace, next_row):
#     # Acquire row
#     cloned_masked_kspace = masked_kspace.clone()
#     # Acquire row for all samples in the batch
#     # shape = (batch_dim, column, row, complex)
#     cloned_masked_kspace[..., next_row, :] = full_kspace[..., next_row, :]
#     zero_filled, mean, std = get_new_zf(cloned_masked_kspace)
#     return zero_filled, mean, std
#
#
# def acquire_new_zf_batch(full_kspace, batch_masked_kspace, batch_next_rows):
#     # shape = batch x 1 x res = col x res = row x 2
#     batch_cloned_masked_kspace = batch_masked_kspace.clone()
#
#     # Acquire row for all samples in the batch
#     # shape = (batch_dim, column, row, complex)
#     for sl, next_row in enumerate(batch_next_rows):
#         batch_cloned_masked_kspace[sl, :, :, next_row, :] = full_kspace[sl, :, :, next_row, :]
#
#     zero_filled, mean, std = get_new_zf(batch_cloned_masked_kspace)
#     return zero_filled, mean, std
#
#
# def acquire_new_zf_exp(k, mk, to_acquire):
#     # Expand masked kspace over channel dimension to prepare for adding all kspace rows to acquire
#     mk_exp = mk.expand(len(to_acquire), -1, -1, -1).clone()  # TODO: .clone() necessary here? Yes?
#     # Acquire row
#     for index, row in enumerate(to_acquire):
#         mk_exp[index, :, row.item(), :] = k[0, :, row.item(), :]
#     # Obtain zero filled image from all len(to_acquire) new kspaces
#     zero_filled_exp, mean_exp, std_exp = get_new_zf(mk_exp)
#     return zero_filled_exp, mean_exp, std_exp
#
#
# def acquire_new_zf_exp_batch(k, mk, to_acquire):
#     # Expand masked kspace over channel dimension to prepare for adding all kspace rows to acquire
#     # TODO: did making the expand --> repeat change really make the difference? Nope, seems to give exact same
#     # mk_exp = mk.expand(-1, to_acquire.size(1), -1, -1, -1).clone()  # TODO: .clone() necessary here? Yes?
#     mk_exp = mk.repeat(1, to_acquire.size(1), 1, 1, 1)
#     # Loop over slices in batch
#     for sl, rows in enumerate(to_acquire):
#         # Loop over indices to acquire
#         for index, row in enumerate(rows):
#             mk_exp[sl, index, :, row.item(), :] = k[sl, 0, :, row.item(), :]
#     # Obtain zero filled image from all new kspaces
#     zero_filled_exp, mean_exp, std_exp = get_new_zf(mk_exp)
#     return zero_filled_exp, mean_exp, std_exp