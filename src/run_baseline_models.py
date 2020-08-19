import logging
import time
import datetime
import random
import argparse
import pathlib
import wandb
from random import choice
from string import ascii_uppercase
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from piq import psnr

from tensorboardX import SummaryWriter

from src.helpers.torch_metrics import ssim

from src.helpers.utils import (add_mask_params, save_json, check_args_consistency, str2bool, str2none)
from src.helpers.data_loading import create_data_loaders
from src.helpers.states import TRAIN_STATE, DEV_STATE, TEST_STATE
from src.recon_models.recon_model_utils import (acquire_new_zf_exp_batch, acquire_new_zf_batch,
                                                recon_model_forward_pass, load_recon_model)
from src.helpers.fastmri_data import DataTransform, SliceData
from src.helpers import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

outputs = defaultdict(lambda: defaultdict(list))


def create_data_range_dict(loader):
    # Locate ground truths of a volume
    gt_vol_dict = {}
    for it, data in enumerate(loader):
        # TODO: Use fname, slice to create state-step-dependent baseline
        # TODO: use fname and initial loop over gt to find data range per fname
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, slice = data
        for i, vol in enumerate(fname):
            if vol not in gt_vol_dict:
                gt_vol_dict[vol] = []
            gt_vol_dict[vol].append(gt[i] * gt_std[i] + gt_mean[i])

    # Find max of a volume
    data_range_dict = {}
    for vol, gts in gt_vol_dict.items():
        # Shape 1 x 1 x 1 x 1
        data_range_dict[vol] = torch.stack(gts).max().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(args.device)
    del gt_vol_dict

    return data_range_dict


def get_psnr(unnorm_recons, gt_exp, data_range):
    # Have to reshape to batch . trajectories x res x res and then reshape back to batch x trajectories x res x res
    # because of psnr implementation
    psnr_recons = torch.clamp(unnorm_recons, 0., 10.).reshape(gt_exp.size(0) * gt_exp.size(1), 1, args.resolution,
                                                              args.resolution).to('cpu')
    psnr_gt = gt_exp.reshape(gt_exp.size(0) * gt_exp.size(1), 1, args.resolution, args.resolution).to('cpu')
    # First duplicate data range over trajectories, then reshape: this to ensure alignment with recon and gt.
    psnr_data_range = data_range.expand(-1, gt_exp.size(1), -1, -1)
    psnr_data_range = psnr_data_range.reshape(gt_exp.size(0) * gt_exp.size(1), 1, 1, 1).to('cpu')
    psnr_scores = psnr(psnr_recons, psnr_gt, reduction='none', data_range=psnr_data_range)
    psnr_scores = psnr_scores.reshape(gt_exp.size(0), gt_exp.size(1))
    return psnr_scores


def get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model, recon, data_range):
    unnorm_recon = recon[:, 0:1, ...] * gt_std + gt_mean  # Back to original scale for metric

    # shape = batch
    base_score = ssim(unnorm_recon, unnorm_gt, size_average=False,
                      data_range=data_range).mean(-1).mean(-1)  # keep channel dim = 1

    res = mask.size(-2)
    batch_acquired_rows = (mask.squeeze(1).squeeze(1).squeeze(-1) == 1)
    acquired_num = batch_acquired_rows[0, :].sum().item()
    tk = res - acquired_num
    batch_train_rows = torch.zeros((mask.size(0), tk)).long().to(args.device)
    for sl, sl_mask in enumerate(mask.squeeze(1).squeeze(1).squeeze(-1)):
        batch_train_rows[sl] = torch.nonzero((sl_mask == 0), as_tuple=False).flatten()

    # Acquire chosen rows, and compute the improvement target for each (batched)
    # shape = batch x rows = k x res x res
    zf_exp, _, _ = acquire_new_zf_exp_batch(kspace, masked_kspace, batch_train_rows)
    # shape = batch . tk x 1 x res x res, so that we can run the forward model for all rows in the batch
    zf_input = zf_exp.view(mask.size(0) * tk, 1, res, res)
    # shape = batch . tk x 2 x res x res
    recons_output = recon_model_forward_pass(args, recon_model, zf_input)
    # shape = batch . tk x 1 x res x res, extract reconstruction to compute target
    recons = recons_output[:, 0:1, ...]
    # shape = batch x tk x res x res
    recons = recons.view(mask.size(0), tk, res, res)
    unnorm_recons = recons * gt_std + gt_mean  # TODO: Normalisation necessary?
    gt_exp = unnorm_gt.expand(-1, tk, -1, -1)
    # scores = batch x tk (channels), base_score = batch x 1
    scores = ssim(unnorm_recons, gt_exp, size_average=False, data_range=data_range).mean(-1).mean(-1)
    impros = scores - base_score
    # target = batch x rows, batch_train_rows and impros = batch x tk
    target = torch.zeros((mask.size(0), res)).to(args.device)
    for j, train_rows in enumerate(batch_train_rows):
        # impros[j, 0] (slice j, row 0 in train_rows[j]) corresponds to the row train_rows[j, 0] = 9
        # (for instance). This means the improvement 9th row in the kspace ordering is element 0 in impros.
        kspace_row_inds, permuted_inds = train_rows.sort()
        target[j, kspace_row_inds] = impros[j, permuted_inds]
    return target


def get_spectral_dist(args, gt, recon, mask):
    # Using kspace obtained from gt here instead of original kspace, since these might differ due to
    # complex abs and normalisation of original kspace.
    # shape = batch x 1 x res x res x 2
    k_gt = transforms.rfft2(gt)
    k_recon = transforms.rfft2(recon)

    # shape = batch x res
    shaped_mask = mask.view(gt.size(0), gt.size(3))
    # shape = batch x num_unacquired_rows
    unacq_rows = [(row_mask == 0).nonzero().flatten() for row_mask in shaped_mask]

    # shape = batch x num_unacq_rows x res x res x 2
    masked_k_gt = torch.zeros(len(unacq_rows), len(unacq_rows[0]), gt.size(2), gt.size(3), 2).to(args.device)
    masked_k_recon = torch.zeros(len(unacq_rows), len(unacq_rows[0]), gt.size(2), gt.size(3), 2).to(args.device)
    # Loop over slices in batch
    for sl, rows in enumerate(unacq_rows):
        # Loop over indices to acquire
        for index, row in enumerate(rows):
            masked_k_gt[sl, index, :, row.item(), :] = k_gt[sl, 0, :, row.item(), :]
            masked_k_recon[sl, index, :, row.item(), :] = k_recon[sl, 0, :, row.item(), :]

    spectral_gt = transforms.ifft2(masked_k_gt)
    spectral_recon = transforms.ifft2(masked_k_recon)

    # Gamma doesn't matter since we're not training and distance are monotonic in squared_norm.
    # We set it so that distances are scaled nicely for our own inspection (also so that not multiple rows get
    # scores of 1 due to machine precision).
    # Currently chosen empirically such that gamma * squared_norm has values ranging from 0.1 to 10.
    gamma = 0.05
    # shape = batch x num_unacq_rows
    squared_norm = torch.sum((spectral_gt - spectral_recon) ** 2, dim=(2, 3, 4))
    closeness = torch.exp(-1 * gamma * squared_norm)
    # we pick the row with the highest score, which should be the row with the largest distance
    distance = 1 - closeness

    # shape = batch x res
    target = torch.zeros((gt.size(0), gt.size(3))).to(args.device)
    for j, rows in enumerate(unacq_rows):
        kspace_row_inds, permuted_inds = rows.sort()
        # permuted_inds here is just a list of indices [ 0, ..., len(rows) = len(unacq_rows[0]) ]
        target[j, kspace_row_inds] = distance[j, permuted_inds]

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
    impro_input = recon_model_forward_pass(args, recon_model, zf)  # TODO: args is global here!
    return impro_input, zf, mean, std, mask, masked_kspace


class StepMaskFunc:
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


def create_avg_oracle_loader(args, step, rows, split):
    mask = StepMaskFunc(step, rows, args.accelerations)

    if split in ['dev', 'val']:
        dev_path = args.data_path / f'{args.challenge}_val'  # combine with dev STATE
        # dev_path = args.data_path / f'{args.challenge}_train_al'  # combine with train STATE and set mult = 1
        mult = 2 if args.sample_rate == 0.04 else 1  # TODO: this is now hardcoded to get more validation samples: fix this
        # mult = 1
        dev_sample_rate = args.sample_rate * mult
        data = SliceData(
            root=dev_path,
            transform=DataTransform(mask, args.resolution, args.challenge, use_seed=True),
            sample_rate=dev_sample_rate,
            challenge=args.challenge,
            acquisition=args.acquisition,
            center_volume=args.center_volume,
            state=DEV_STATE
        )
    elif split == 'test':
        test_path = args.data_path / f'{args.challenge}_test_al'  # combine with dev STATE

        data = SliceData(
            root=test_path,
            transform=DataTransform(mask, args.resolution, args.challenge, use_seed=True),
            sample_rate=args.sample_rate,
            challenge=args.challenge,
            acquisition=args.acquisition,
            center_volume=args.center_volume,
            state=TEST_STATE
        )
    elif split == 'train':
        train_path = args.data_path / f'{args.challenge}_train_al'  # combine with dev STATE

        data = SliceData(
            root=train_path,
            transform=DataTransform(mask, args.resolution, args.challenge, use_seed=True),
            sample_rate=args.sample_rate,
            challenge=args.challenge,
            acquisition=args.acquisition,
            center_volume=args.center_volume,
            state=TRAIN_STATE
        )
    else:
        raise ValueError()

    loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader


def run_average_oracle(args, recon_model):
    epoch_outputs = defaultdict(list)
    start = time.perf_counter()

    rows = []
    ssims = np.array([0. for _ in range(args.acquisition_steps + 1)])
    psnrs = np.array([0. for _ in range(args.acquisition_steps + 1)])
    with torch.no_grad():
        for step in range(args.acquisition_steps + 1):
            train_loader, dev_loader, test_loader, _ = create_data_loaders(args, shuffle_train=False)
            # Loader for this step: includes starting rows and best rows from previous steps in mask
            if args.data_split in ['dev', 'val']:
                loader = create_avg_oracle_loader(args, step, rows, 'dev')
            elif args.data_split == 'test':
                loader = create_avg_oracle_loader(args, step, rows, 'test')
            elif args.data_split == 'train':
                loader = create_avg_oracle_loader(args, step, rows, 'train')
            else:
                raise ValueError

            if args.data_range == 'gt':
                data_range_dict = None
            elif args.data_range == 'volume':
                data_range_dict = create_data_range_dict(loader)
            else:
                raise ValueError(f'{args.data_range} is not valid')

            if args.model_type == 'oracle_average' or (args.model_type == 'notime_oracle_average' and step == 0):
                sum_impros = 0.
                tbs = 0.

            # Find average best improvement over dataset for this step
            for it, data in enumerate(loader):
                kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
                # TODO: Maybe normalisation unnecessary for SSIM target?
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
                if args.data_range == 'gt':
                    data_range = unnorm_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
                elif args.data_range == 'volume':
                    data_range = torch.stack([data_range_dict[vol] for vol in fname])
                else:
                    raise ValueError(f'{args.data_range} is not valid')

                # Base reconstruction model forward pass
                recon = recon_model_forward_pass(args, recon_model, zf)

                unnorm_recon = recon[:, 0:1, :, :] * gt_std + gt_mean
                ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                data_range=data_range).mean(dim=(-1, -2)).sum()
                psnr_val = psnr(torch.clamp(unnorm_recon, 0., 10.).to('cpu'),
                                unnorm_gt.to('cpu'),
                                reduction='none',
                                data_range=data_range.to('cpu')).sum()
                ssims[step] += ssim_val.item()
                psnrs[step] += psnr_val.item()

                if args.model_type == 'oracle_average' or (args.model_type == 'notime_oracle_average' and step == 0):
                    tbs += mask.size(0)
                    if step != args.acquisition_steps:  # still acquire, otherwise just need final value, no acquisition
                        output = get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                            recon, data_range)
                        output = output.to('cpu').numpy()
                        sum_impros += output.sum(axis=0)
                        epoch_outputs[step + 1].append(output)
                else:
                    if step != args.acquisition_steps:
                        epoch_outputs[step + 1].append(output)

            if args.model_type == 'oracle_average':
                if step != args.acquisition_steps:  # still acquire, otherwise just need final value, no acquisition
                    rows.append(np.argmax(sum_impros / tbs))
                    print(tbs)

            if args.model_type == 'notime_oracle_average':
                if step != args.acquisition_steps:
                    # Get next highest value from sum_impros
                    # TODO: Could make more efficient by just using these values as ssim_scores and skipping the rest
                    rows.append(np.argsort(sum_impros / tbs)[-1 - step])
                    print(tbs)

    ssims /= tbs
    psnrs /= tbs

    for step in range(args.acquisition_steps):
        outputs[-1][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
    save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

    return ssims, psnrs, time.perf_counter() - start


def run_oracle(args, recon_model, dev_loader, data_range_dict):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """

    ssims = 0
    psnrs = 0
    epoch_outputs = defaultdict(list)
    start = time.perf_counter()
    tbs = 0
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
            # logging.info('Batch {}/{}'.format(it + 1, len(dev_loader)))
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
            # TODO: Maybe normalisation unnecessary for SSIM target?
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
            if args.data_range == 'gt':
                data_range = unnorm_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            elif args.data_range == 'volume':
                data_range = torch.stack([data_range_dict[vol] for vol in fname])
            else:
                raise ValueError(f'{args.data_range} is not valid')

            tbs += mask.size(0)

            # Base reconstruction model forward pass
            recon = recon_model_forward_pass(args, recon_model, zf)

            unnorm_recon = recon[:, 0:1, :, :] * gt_std + gt_mean
            init_ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                 data_range=data_range).mean(dim=(-1, -2)).sum()
            init_psnr_val = psnr(torch.clamp(unnorm_recon, 0., 10.).to('cpu'),
                                 unnorm_gt.to('cpu'),
                                 reduction='none',
                                 data_range=data_range.to('cpu')).sum()
            # init_ssim_val = ssim(recon[:, 0:1, :, :], gt, size_average=False,
            #                      data_range=data_range).mean(dim=(-1, -2)).sum()
            batch_ssims = [init_ssim_val.item()]
            batch_psnrs = [init_psnr_val.item()]

            for step in range(args.acquisition_steps):
                if args.model_type == 'oracle':
                    output = get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                        recon, data_range)
                elif args.model_type == 'notime_oracle':
                    if step == 0:
                        output = get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                            recon, data_range)
                    else:
                        # Don't recompute output
                        output = output
                        # Set current largest value very low, so that next largest value is now largest
                        output[:, torch.argmax(output, dim=1)] = -1e-3
                elif args.model_type == 'center_sym':  # Always get high score for most center unacquired row
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[(mask.size(-2) - 1) / 2 - abs(i - 0.1 - (mask.size(-2) - 1) / 2)
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'center_asym_left':
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[i if i <= mask.size(-2) // 2 else 0
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'center_asym_right':
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[mask.size(-2) - i if i >= mask.size(-2) // 2 else 0
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'random':  # Generate random scores (set acquired to 0. to perform filtering)
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.randn((mask.size(0), mask.size(-2)))
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'equispace_twosided':
                    interval = int(args.resolution * (1 - 1 / args.accelerations[0]))
                    output = torch.zeros((mask.size(0), mask.size(-2)))
                    equi = interval // args.acquisition_steps
                    # Something like this
                    if step <= args.acquisition_steps // 2:
                        select = (step + 1) * equi - 1
                    else:
                        select = args.resolution - (step - args.acquisition_steps // 2) * equi
                    output[:, select] = 1.
                    output = output.to(args.device)
                elif args.model_type == 'equispace_onesided':
                    interval = int(args.resolution * (1 - 1 / args.accelerations[0]) / 2)
                    output = torch.zeros((mask.size(0), mask.size(-2)))
                    equi = interval // args.acquisition_steps
                    select = (step + 1) * equi - 1
                    output[:, select] = 1.
                    output = output.to(args.device)
                elif args.model_type == 'spectral':
                    # K-space similarity model proxy from Zhang et al. (2019)
                    # Instead of training an evaluator to determine kspace distance, we calculate ground truth
                    # kspace distances between the reconstruction and ground truth dev data, using the same distance
                    # metric. This is then used to guide acquisitions.
                    output = get_spectral_dist(args, gt, recon, mask)

                epoch_outputs[step + 1].append(output.to('cpu').numpy())
                # Greedy policy (size = batch)
                _, next_rows = torch.max(output, dim=1)

                # Acquire this row
                impro_input, zf, _, _, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                         recon_model)
                unnorm_recon = impro_input[:, 0:1, :, :] * gt_std + gt_mean
                # shape = 1
                ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                data_range=data_range).mean(dim=(-1, -2)).sum()
                # ssim_val = ssim(impro_input[:, 0:1, :, :], gt, size_average=False,
                #                 data_range=data_range).mean(dim=(-1, -2)).sum()
                # Clamp to min 0 because PSNR involves logs and reconstruction sometimes gives slightly negative values
                # when trying to match a near-zero value. How does fastMRI deal with this?
                psnr_val = psnr(torch.clamp(unnorm_recon, 0., 10.).to('cpu'),
                                unnorm_gt.to('cpu'),
                                reduction='none',
                                data_range=data_range.to('cpu')).sum()

                # eventually shape = al_steps
                batch_ssims.append(ssim_val.item())
                batch_psnrs.append(psnr_val.item())

            # shape = al_steps
            ssims += np.array(batch_ssims)
            psnrs += np.array(batch_psnrs)
    ssims /= tbs
    psnrs /= tbs

    # for step in range(args.acquisition_steps):
    #     outputs[-1][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
    # save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

    return ssims, psnrs, time.perf_counter() - start


def main(args):
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    check_args_consistency(args, recon_args)

    # Add mask parameters for training
    args = add_mask_params(args, recon_args)

    # Create directory to store results in
    savestr = 'res{}_al{}_accel{}_{}_{}_{}_{}'.format(args.resolution, args.acquisition_steps,
                                                      args.accelerations, args.model_type, args.recon_model_name,
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

    if args.model_type in ['oracle_average', 'notime_oracle_average']:
        oracle_ssims, oracle_psnrs, oracle_time = run_average_oracle(args, recon_model)
    else:
        # Create data loaders
        train_loader, dev_loader, test_loader, _ = create_data_loaders(args, shuffle_train=True)

        if args.data_range == 'gt':
            data_range_dict = None
        elif args.data_range == 'volume':
            if args.data_split in ['dev', 'val']:
                data_range_dict = create_data_range_dict(dev_loader)
            elif args.data_split == 'test':
                data_range_dict = create_data_range_dict(test_loader)
            elif args.data_split == 'train':
                data_range_dict = create_data_range_dict(train_loader)
            else:
                raise ValueError
        else:
            raise ValueError(f'{args.data_range} is not valid')

        # # TODO: remove this
        # train_batch = next(iter(train_loader))
        # train_loader = [train_batch] * 1
        # dev_batch = next(iter(dev_loader))
        # dev_loader = [dev_batch] * 1
        # # dev_loader = train_loader

        if args.data_split in ['dev', 'val']:
            oracle_ssims, oracle_psnrs, oracle_time = run_oracle(args, recon_model, dev_loader, data_range_dict)
        elif args.data_split == 'test':
            oracle_ssims, oracle_psnrs, oracle_time = run_oracle(args, recon_model, test_loader, data_range_dict)
        elif args.data_split == 'train':
            oracle_ssims, oracle_psnrs, oracle_time = run_oracle(args, recon_model, train_loader, data_range_dict)
        else:
            raise ValueError

    ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(oracle_ssims)])
    psnrs_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(oracle_psnrs)])
    logging.info(f'  DevSSIM = [{ssims_str}]')
    logging.info(f'  DevPSNR = [{psnrs_str}]')
    logging.info(f'  Time = {oracle_time:.2f}s')

    # For storing in wandb
    for epoch in range(args.num_epochs + 1):
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] ')
        if args.model_type == 'random':
            if args.data_split in ['dev', 'val']:
                oracle_ssims, oracle_psnrs, oracle_time = run_oracle(args, recon_model, dev_loader, data_range_dict)
            elif args.data_split == 'test':
                oracle_ssims, oracle_psnrs, oracle_time = run_oracle(args, recon_model, test_loader, data_range_dict)
            else:
                oracle_ssims, oracle_psnrs, oracle_time = run_oracle(args, recon_model, train_loader, data_range_dict)
            ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(oracle_ssims)])
            psnrs_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(oracle_psnrs)])
            logging.info(f'DevSSIMTime = {oracle_time:.2f}s')
        logging.info(f'  DevSSIM = [{ssims_str}]')
        logging.info(f'  DevPSNR = [{psnrs_str}]')
        if args.wandb:
            if args.data_split in ['val', 'test']:
                wandb.log({'val_ssims': {str(key): val for key, val in enumerate(oracle_ssims)}}, step=epoch)
                wandb.log({'val_psnrs': {str(key): val for key, val in enumerate(oracle_psnrs)}}, step=epoch)
            elif args.data_split == 'train':
                wandb.log({'train_ssims': {str(key): val for key, val in enumerate(oracle_ssims)}}, step=epoch)
                wandb.log({'train_psnrs': {str(key): val for key, val in enumerate(oracle_psnrs)}}, step=epoch)
            else:
                raise ValueError()
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--dataset', choices=['fastmri', 'cifar10'], required=True,
                        help='Dataset to use.')
    parser.add_argument('--wandb',  action='store_true',
                        help='Whether to use wandb logging for this run.')

    parser.add_argument('--model-type', choices=['center_sym', 'center_asym_left', 'center_asym_right',
                                                 'random', 'oracle', 'oracle_average', 'spectral',
                                                 'notime_oracle', 'notime_oracle_average', 'equispace_onesided',
                                                 'equispace_twosided'], required=True,
                        help='Type of model to use.')
    parser.add_argument('--data-range', type=str, default='volume',
                        help="Type of data range to use. Options are 'volume' and 'gt'.")

    # Data parameters
    parser.add_argument('--challenge', type=str, default='singlecoil',
                        help='Which challenge for fastMRI training.')
    parser.add_argument('--data-path', type=pathlib.Path, default=None,
                        help='Path to the dataset. Required for fastMRI training.')
    parser.add_argument('--sample-rate', type=float, default=1.,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str2none, default='CORPD_FBK',
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')

    # Reconstruction model
    parser.add_argument('--recon-model-checkpoint', type=pathlib.Path, default=None,
                        help='Path to a pretrained reconstruction model. If None then recon-model-name should be'
                        'set to zero_filled.')
    parser.add_argument('--recon-model-name', choices=['kengal_laplace', 'kengal_gauss', 'zero_filled', 'nounc'],
                        required=True, help='Reconstruction model name corresponding to model checkpoint.')

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
    parser.add_argument('--use-data-state', type=str2bool, default=False,
                        help='Whether to use fixed data state for random data selection.')
    parser.add_argument('--data-split', type=str, default='dev',
                        help='Which data split to use.')

    parser.add_argument('--project',  type=str, default='mrimpro',
                        help='Wandb project name to use.')
    parser.add_argument('--original_setting', type=str2bool, default=True,
                        help='Whether to use original data setting used for knee experiments.')
    parser.add_argument('--low_res', type=str2bool, default=False,
                        help='Whether to use a low res full image, rather than a high res small image when cropping.')

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

    if args.use_data_state:
        args.train_state = TRAIN_STATE
        args.dev_state = DEV_STATE
        args.test_state = TEST_STATE
    else:
        args.train_state = None
        args.dev_state = None
        args.test_state = None
    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False
    main(args)
