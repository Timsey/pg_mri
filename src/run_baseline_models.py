import logging
import time
import datetime
import random
import argparse
import pathlib
import wandb
from random import choice
from string import ascii_uppercase

import numpy as np
import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from src.helpers.torch_metrics import compute_ssim, compute_psnr
from src.helpers.utils import add_mask_params, save_json, str2bool, str2none
from src.helpers.data_loading import create_data_loader, SliceData, DataTransform
from src.reconstruction_model.reconstruction_model_utils import load_recon_model
from src.policy_model.policy_model_utils import create_data_range_dict, compute_next_step_reconstruction, compute_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_all_scores(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model, data_range):
    output = torch.zeros((kspace.shape[0], kspace.shape[-2]))
    # Set all unacquired rows as rows to acquire
    to_acquire = [[] for _ in range(kspace.shape[0])]
    unacquired_inds = (mask.squeeze(1).squeeze(1).squeeze(-1) == 0).nonzero(as_tuple=False)
    for sl, ind in unacquired_inds:
        to_acquire[sl].append(ind)
    to_acquire = torch.tensor(to_acquire)
    _, _, _, recon = compute_next_step_reconstruction(recon_model, kspace, masked_kspace, mask, to_acquire)
    ssim_scores = compute_scores(args, recon, gt_mean, gt_std, unnorm_gt, data_range, comp_psnr=False)
    old_slice, idx = -1, -1
    for sl, ind in unacquired_inds:
        if sl == old_slice:
            idx += 1
        else:
            idx = 0
            old_slice = sl
        output[sl, ind] = ssim_scores[sl, idx]
    return output


class StepMaskFunc:
    # Mask function for average_oracle
    def __init__(self, step, rows, accelerations):
        assert len(rows) == step, 'Mismatch between step and number of acquired rows'
        self.step = step
        self.rows = rows
        assert len(accelerations) == 1, "StepMaskFunc only works for a single acceleration at a time"
        self.acceleration = accelerations[0]
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        num_cols = shape[-2]

        # Create the mask
        num_low_freqs = num_cols // self.acceleration
        mask = np.zeros(num_cols)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True
        mask[self.rows] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def create_avg_oracle_loader(args, step, rows):
    mask = StepMaskFunc(step, rows, args.accelerations)

    # TODO: Fix these paths!
    if args.partition == 'train':
        path = args.data_path / f'singlecoil_train_al'
    elif args.partition == 'val':
        path = args.data_path / f'singlecoil_val'
    elif args.partition == 'test':
        path = args.data_path / f'singlecoil_test_al'

    dataset = SliceData(
        root=path,
        transform=DataTransform(mask, args.resolution, use_seed=True),
        dataset=args.dataset,
        sample_rate=args.sample_rate,
        acquisition=args.acquisition,
        center_volume=args.center_volume
    )

    print(f'{args.partition.capitalize()} slices: {len(dataset)}')

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader


def run_average_oracle(args, recon_model):
    start = time.perf_counter()
    rows = []
    ssims = np.array([0. for _ in range(args.acquisition_steps + 1)])
    psnrs = np.array([0. for _ in range(args.acquisition_steps + 1)])
    with torch.no_grad():
        for step in range(args.acquisition_steps + 1):
            # Loader for this step: includes starting rows and best rows from previous steps in mask
            loader = create_avg_oracle_loader(args, step, rows)
            data_range_dict = create_data_range_dict(args, loader)
            sum_impros = 0.
            tbs = 0.
            # Find average best improvement over dataset for this step
            for it, data in enumerate(loader):
                kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
                # shape after unsqueeze = batch x channel x columns x rows x complex
                kspace = kspace.unsqueeze(1).to(args.device)
                masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
                mask = mask.unsqueeze(1).to(args.device)
                # shape after unsqueeze = batch x channel x columns x rows
                zf = zf.unsqueeze(1).to(args.device)
                gt = gt.unsqueeze(1).to(args.device)
                gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
                gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
                unnorm_gt = gt * gt_std + gt_mean
                data_range = torch.stack([data_range_dict[vol] for vol in fname])

                # Base reconstruction model forward pass
                recon = recon_model(zf)
                unnorm_recon = recon * gt_std + gt_mean
                ssim_val = compute_ssim(unnorm_recon, unnorm_gt, size_average=False,
                                        data_range=data_range).mean(dim=(-1, -2)).sum()
                psnr_val = compute_psnr(args, unnorm_recon, unnorm_gt, data_range).sum()
                ssims[step] += ssim_val.item()
                psnrs[step] += psnr_val.item()

                tbs += mask.size(0)
                if step != args.acquisition_steps:  # 'output' is required for acquisition
                    output = compute_all_scores(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std,
                                                recon_model, data_range)
                    output = output.to('cpu').numpy()
                    sum_impros += output.sum(axis=0)  # sum of ssim_scores over slices for each measurement

            if step != args.acquisition_steps:  # still acquire, otherwise just need final value, no acquisition
                rows.append(np.argmax(sum_impros / tbs))

    ssims /= tbs
    psnrs /= tbs

    return ssims, psnrs, time.perf_counter() - start


def run_baseline(args, recon_model, loader, data_range_dict):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """

    ssims = 0
    psnrs = 0
    start = time.perf_counter()
    tbs = 0
    with torch.no_grad():
        for it, data in enumerate(loader):
            # logging.info('Batch {}/{}'.format(it + 1, len(loader)))
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
            # shape after unsqueeze = batch x channel x columns x rows x complex
            kspace = kspace.unsqueeze(1).to(args.device)
            masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
            # shape after unsqueeze = batch x channel x columns x rows
            zf = zf.unsqueeze(1).to(args.device)
            gt = gt.unsqueeze(1).to(args.device)
            gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            unnorm_gt = gt * gt_std + gt_mean
            data_range = torch.stack([data_range_dict[vol] for vol in fname])
            tbs += mask.size(0)

            # Base reconstruction model forward pass
            recon = recon_model(zf)
            unnorm_recon = recon * gt_std + gt_mean
            init_ssim_val = compute_ssim(unnorm_recon, unnorm_gt, size_average=False,
                                         data_range=data_range).mean(dim=(-1, -2)).sum()
            init_psnr_val = compute_psnr(args, unnorm_recon, unnorm_gt, data_range).sum()
            batch_ssims = [init_ssim_val.item()]
            batch_psnrs = [init_psnr_val.item()]

            for step in range(args.acquisition_steps):
                if args.model_type == 'oracle':
                    output = compute_all_scores(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std,
                                                recon_model, data_range)
                elif args.model_type == 'random':  # Generate random scores (set acquired to 0. to perform filtering)
                    acquired = mask.squeeze().nonzero(as_tuple=False)
                    output = torch.randn((mask.size(0), mask.size(-2)))
                    output[acquired[:, 0], acquired[:, 1]] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'equispace_twosided':
                    interval = int(args.resolution * (1 - 1 / args.accelerations[0]))
                    output = torch.zeros((mask.size(0), mask.size(-2)))
                    equi = interval / args.acquisition_steps
                    # Something like this: only tested for even acceleration and acquisition_steps
                    if step < args.acquisition_steps // 2:
                        select = int((step + 1) * equi - 1)
                    else:
                        select = int(args.resolution - (step + 1 - args.acquisition_steps // 2) * equi)
                    output[:, select] = 1.
                    output = output.to(args.device)
                elif args.model_type == 'equispace_onesided':
                    interval = int(args.resolution * (1 - 1 / args.accelerations[0]) / 2)
                    output = torch.zeros((mask.size(0), mask.size(-2)))
                    equi = interval / args.acquisition_steps
                    # Something like this: only tested for even acceleration and acquisition_steps
                    select = int((step + 1) * equi - 1)
                    output[:, select] = 1.
                    output = output.to(args.device)

                # Greedy policy on computed targets (size = batch)
                actions = torch.max(output, dim=1, keepdim=True)[1]
                # Acquire this measurement
                mask, masked_kspace, zf, recons = compute_next_step_reconstruction(recon_model, kspace,
                                                                                   masked_kspace, mask, actions)
                unnorm_recon = recons * gt_std + gt_mean
                ssim_val = compute_ssim(unnorm_recon, unnorm_gt, size_average=False,
                                        data_range=data_range).mean(dim=(-1, -2)).sum()
                psnr_val = compute_psnr(args, unnorm_recon, unnorm_gt, data_range).sum()
                # eventually shape = al_steps
                batch_ssims.append(ssim_val.item())
                batch_psnrs.append(psnr_val.item())

            ssims += np.array(batch_ssims)
            psnrs += np.array(batch_psnrs)

    ssims /= tbs
    psnrs /= tbs

    return ssims, psnrs, time.perf_counter() - start


def main(args):
    # For consistency
    args.val_batch_size = args.batch_size
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    # Add mask parameters for training
    args = add_mask_params(args)

    # Create directory to store results in
    savestr = '{}_res{}_al{}_accel{}_{}_{}_{}'.format(args.dataset, args.resolution, args.acquisition_steps,
                                                      args.accelerations, args.model_type,
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                      ''.join(choice(ascii_uppercase) for _ in range(5)))
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

    if args.model_type == 'average_oracle':
        baseline_ssims, baseline_psnrs, baseline_time = run_average_oracle(args, recon_model)
    else:
        # Create data loader
        loader = create_data_loader(args, args.partition)
        data_range_dict = create_data_range_dict(args, loader)
        baseline_ssims, baseline_psnrs, baseline_time = run_baseline(args, recon_model, loader, data_range_dict)

    # Logging
    ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(baseline_ssims)])
    psnrs_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(baseline_psnrs)])
    logging.info(f'  SSIM = [{ssims_str}]')
    logging.info(f'  PSNR = [{psnrs_str}]')
    logging.info(f'  Time = {baseline_time:.2f}s')

    # For storing in wandb
    for epoch in range(args.num_epochs + 1):
        if args.wandb:
            wandb.log({f'{args.partition}_ssims': {str(key): val for key, val in enumerate(baseline_ssims)}}, step=epoch)
            wandb.log({f'{args.partition}_psnrs': {str(key): val for key, val in enumerate(baseline_psnrs)}}, step=epoch)

    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators. '
                                                            'Set to 0 to use random seed.')
    parser.add_argument('--resolution', default=128, type=int, help='Resolution of images')
    parser.add_argument('--dataset', choices=['knee', 'brain'], default='knee',
                        help='Dataset to use.')
    parser.add_argument('--wandb', type=str2bool, default=False,
                        help='Whether to use wandb logging for this run.')
    parser.add_argument('--model_type', choices=['random', 'oracle', 'average_oracle', 'equispace_onesided',
                                                 'equispace_twosided'], required=True,
                        help='Type of baseline to run.')

    parser.add_argument('--data_path', type=pathlib.Path, default=None,
                        help='Path to the dataset. Required for fastMRI training.')
    parser.add_argument('--sample_rate', type=float, default=0.5,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str2none, default=None,
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')
    parser.add_argument('--recon_model_checkpoint', type=pathlib.Path, required=True,
                        help='Path to a pretrained reconstruction model.')
    parser.add_argument('--center_volume', type=str2bool, default=True,
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--reciprocals_in_center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')
    parser.add_argument('--acquisition_steps', default=16, type=int, help='Acquisition steps to train for per image.')
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size for training set.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to run the baseline for.')
    parser.add_argument('--data_parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, required=True,
                        help='Directory where results should be saved. Will create a timestamped folder '
                        'in provided directory each run.')
    parser.add_argument('--partition', type=str, choices=['train', 'val', 'test'], default='val',
                        help='Which data split to use.')
    parser.add_argument('--project', type=str2none, default=None,
                        help='Wandb project name to use.')

    return parser


if __name__ == '__main__':
    # To fix known issue with h5py + multiprocessing
    # See: https://discuss.pytorch.org/t/incorrect-data-using-h5py-with-dataloader/7079/2?u=ptrblck
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    args = create_arg_parser().parse_args()
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    if args.wandb:
        wandb.init(project=args.project, config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False
    main(args)
