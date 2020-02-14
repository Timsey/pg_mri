import logging
import time
import datetime
import random
import argparse
import pathlib
import wandb
from collections import defaultdict

import numpy as np
import torch

from tensorboardX import SummaryWriter

from src.helpers.torch_metrics import ssim

from src.helpers.utils import (add_mask_params, save_json, check_args_consistency)
from src.helpers.data_loading import create_data_loaders
from src.recon_models.recon_model_utils import (acquire_new_zf_exp_batch, acquire_new_zf_batch,
                                                recon_model_forward_pass, load_recon_model)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

outputs = defaultdict(lambda: defaultdict(list))


def get_target(args, kspace, masked_kspace, mask, gt, mean, std, recon_model, recon):
    norm_recon = recon[:, 0:1, ...] * std + mean  # Back to original scale for metric

    # shape = batch
    base_score = ssim(norm_recon, gt, size_average=False, data_range=1e-4).mean(-1).mean(-1)  # keep channel dim = 1

    res = mask.size(-2)
    batch_acquired_rows = (mask.squeeze(1).squeeze(1).squeeze(-1) == 1)
    acquired_num = batch_acquired_rows[0, :].sum().item()
    tk = res - acquired_num
    batch_train_rows = torch.zeros((mask.size(0), tk)).long().to(args.device)
    for sl, sl_mask in enumerate(mask.squeeze(1).squeeze(1).squeeze(-1)):
        batch_train_rows[sl] = (sl_mask == 0).nonzero().flatten()

    # Acquire chosen rows, and compute the improvement target for each (batched)
    # shape = batch x rows = k x res x res
    zf_exp, mean_exp, std_exp = acquire_new_zf_exp_batch(kspace, masked_kspace, batch_train_rows)
    # shape = batch . tk x 1 x res x res, so that we can run the forward model for all rows in the batch
    zf_input = zf_exp.view(mask.size(0) * tk, 1, res, res)
    # shape = batch . tk x 2 x res x res
    recons_output = recon_model_forward_pass(args, recon_model, zf_input)
    # shape = batch . tk x 1 x res x res, extract reconstruction to compute target
    recons = recons_output[:, 0:1, ...]
    # shape = batch x tk x res x res
    recons = recons.view(mask.size(0), tk, res, res)
    norm_recons = recons * std_exp + mean_exp  # TODO: Normalisation necessary?
    gt = gt.expand(-1, tk, -1, -1)
    # scores = batch x tk (channels), base_score = batch x 1
    scores = ssim(norm_recons, gt, size_average=False, data_range=1e-4).mean(-1).mean(-1)
    impros = scores - base_score
    # target = batch x rows, batch_train_rows and impros = batch x tk
    target = torch.zeros((mask.size(0), res)).to(args.device)
    for j, train_rows in enumerate(batch_train_rows):
        # impros[j, 0] (slice j, row 0 in train_rows[j]) corresponds to the row train_rows[j, 0] = 9
        # (for instance). This means the improvement 9th row in the kspace ordering is element 0 in impros.
        kspace_row_inds, permuted_inds = train_rows.sort()
        target[j, kspace_row_inds] = impros[j, permuted_inds]
    return target


def acquire_row(kspace, masked_kspace, next_rows, mask, recon_model):
    zf, mean, std = acquire_new_zf_batch(kspace, masked_kspace, next_rows)
    # Don't forget to change mask for impro_model (necessary if impro model uses mask)
    # Also need to change masked kspace for recon model (getting correct next-step zf)
    # TODO: maybe do this in the acquire_new_zf_batch() function. Doesn't fit with other functions of same
    #  description, but this one is particularly used for this acquisition loop.
    for sl, next_row in enumerate(next_rows):
        mask[sl, :, :, next_row, :] = 1.
        masked_kspace[sl, :, :, next_row, :] = kspace[sl, :, :, next_row, :]
    # Get new reconstruction for batch
    impro_input = recon_model_forward_pass(args, recon_model, zf)
    return impro_input, zf, mean, std, mask, masked_kspace


def run_oracle(args, recon_model, dev_loader):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """

    ssims = 0
    epoch_outputs = defaultdict(list)
    start = time.perf_counter()
    tbs = 0
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
            # logging.info('Batch {}/{}'.format(it + 1, len(dev_loader)))
            kspace, masked_kspace, mask, zf, gt, mean, std, fname, slices = data
            tbs += mask.size(0)
            # shape after unsqueeze = batch x channel x columns x rows x complex
            kspace = kspace.unsqueeze(1).to(args.device)
            masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
            # shape after unsqueeze = batch x channel x columns x rows
            zf = zf.unsqueeze(1).to(args.device)
            gt = gt.unsqueeze(1).to(args.device)
            mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            gt = gt * std + mean

            # Base reconstruction model forward pass
            recon = recon_model_forward_pass(args, recon_model, zf)

            norm_recon = recon[:, 0:1, :, :] * std + mean
            init_ssim_val = ssim(norm_recon, gt, size_average=False, data_range=1e-4).mean(dim=(-1, -2)).sum()

            batch_ssims = [init_ssim_val.item()]

            for step in range(args.acquisition_steps):
                if args.model_type == 'oracle':
                    output = get_target(args, kspace, masked_kspace, mask, gt, mean, std, recon_model, recon)
                elif args.model_type == 'center':  # Always get high score for most center unacquired row
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[(mask.size(-2) - 1) / 2 - abs(i - 0.1 - (mask.size(-2) - 1) / 2)
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'random':  # Generate random scores (set acquired to 0. to perform filtering)
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.randn((mask.size(0), mask.size(-2)))
                    output[:, acquired] = 0.
                    output = output.to(args.device)

                epoch_outputs[step + 1].append(output.to('cpu').numpy())
                # Greedy policy (size = batch)
                _, next_rows = torch.max(output, dim=1)

                # Acquire this row
                impro_input, zf, mean, std, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                              recon_model)
                norm_recon = impro_input[:, 0:1, :, :] * std + mean
                # shape = 1
                ssim_val = ssim(norm_recon, gt, size_average=False, data_range=1e-4).mean(dim=(-1, -2)).sum()
                # eventually shape = al_steps
                batch_ssims.append(ssim_val.item())

            # shape = al_steps
            ssims += np.array(batch_ssims)

    ssims /= tbs

    for step in range(args.acquisition_steps):
        outputs[-1][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
    save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

    return ssims, time.perf_counter() - start


def main(args):
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    check_args_consistency(args, recon_args)

    # Add mask parameters for training
    args = add_mask_params(args, recon_args)

    start_epoch = 0
    # Create directory to store results in
    savestr = 'res{}_al{}_accel{}_{}_{}_{}'.format(args.resolution, args.acquisition_steps, args.accelerations,
                                                   args.model_type, args.recon_model_name,
                                                   datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    args.run_dir = args.exp_dir / savestr
    args.run_dir.mkdir(parents=True, exist_ok=False)

    if args.wandb:
        wandb.config.update(args)

    # Logging
    logging.info(args)
    logging.info(recon_model)
    logging.info('Model type: {}'.format(args.model_type))

    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    # Create data loaders
    _, dev_loader, _, _ = create_data_loaders(args)

    oracle_dev_ssims, oracle_time = run_oracle(args, recon_model, dev_loader)

    dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(oracle_dev_ssims)])
    logging.info(f'  DevSSIM = [{dev_ssims_str}]')
    logging.info(f'DevSSIMTime = {oracle_time:.2f}s')

    # For storing in wandb
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] ')
        if args.model_type == 'random':
            oracle_dev_ssims, oracle_time = run_oracle(args, recon_model, dev_loader)
            dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(oracle_dev_ssims)])
            logging.info(f'DevSSIMTime = {oracle_time:.2f}s')
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        if args.wandb:
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(oracle_dev_ssims)}}, step=epoch + 1)
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--dataset', choices=['fastmri', 'cifar10'], required=True,
                        help='Dataset to use.')
    parser.add_argument('--wandb',  action='store_true',
                        help='Whether to use wandb logging for this run.')

    parser.add_argument('--model-type', choices=['center', 'random', 'oracle'], required=True,
                        help='Type of model to use.')

    # Data parameters
    parser.add_argument('--challenge', type=str, default='singlecoil',
                        help='Which challenge for fastMRI training.')
    parser.add_argument('--data-path', type=pathlib.Path, default=None,
                        help='Path to the dataset. Required for fastMRI training.')
    parser.add_argument('--sample-rate', type=float, default=1.,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str, default='CORPD_FBK',
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')

    # Reconstruction model
    parser.add_argument('--recon-model-checkpoint', type=pathlib.Path, default=None,
                        help='Path to a pretrained reconstruction model. If None then recon-model-name should be'
                        'set to zero_filled.')
    parser.add_argument('--recon-model-name', choices=['kengal_laplace', 'kengal_gauss', 'zero_filled'], required=True,
                        help='Reconstruction model name corresponding to model checkpoint.')
    parser.add_argument('--in-chans', default=1, type=int, help='Number of image input channels')

    parser.add_argument('--center-volume', action='store_true',
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--use-recon-mask-params', action='store_true',
                        help='Whether to use mask parameter settings (acceleration and center fraction) that the '
                        'reconstruction model was trained on. This will overwrite any other mask settings.')

    # Mask parameters, preferably they match the parameters the reconstruction model was trained on. Also see
    # argument use-recon-mask-params above.
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--reciprocals-in-center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')

    parser.add_argument('--acquisition-steps', default=10, type=int, help='Acquisition steps to train for per image.')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')

    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, required=True,
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                        'in provided directory each run')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Set verbosity level. Lowest=0, highest=4."')
    return parser


if __name__ == '__main__':
    # To fix known issue with h5py + multiprocessing
    # See: https://discuss.pytorch.org/t/incorrect-data-using-h5py-with-dataloader/7079/2?u=ptrblck
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    if args.wandb:
        wandb.init(project='mrimpro', config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False
    main(args)
