"""
Script for running sweeps on DAS5.

Run as follows in mrimpro base dir:

> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages wandb sweep src/RL_sweep.yaml
> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages wandb agent timsey/mrimpro/SWEEP_ID

"""


import logging
import time
import copy
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
from tensorboardX import SummaryWriter
from piq import psnr

import sys
# sys.path.insert(0, '/home/timsey/Projects/mrimpro/')  # noqa: F401
# sys.path.insert(0, '/Users/tbakker/Projects/mrimpro/')  # noqa: F401
sys.path.insert(0, '/var/scratch/tbbakker/mrimpro/')  # noqa: F401

from src.helpers.torch_metrics import ssim
from src.helpers.utils import (add_mask_params, save_json, check_args_consistency, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters, str2bool, str2none)
from src.helpers.data_loading import create_data_loaders
from src.helpers.states import DEV_STATE, TRAIN_STATE, TEST_STATE
from src.recon_models.recon_model_utils import (get_new_zf, acquire_new_zf_batch, create_impro_model_input,
                                                load_recon_model)
from src.impro_models.impro_model_utils import (build_impro_model, load_impro_model, build_optim, save_model,
                                                impro_model_forward_pass)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

targets = defaultdict(lambda: defaultdict(lambda: 0))
target_counts = defaultdict(lambda: defaultdict(lambda: 0))
outputs = defaultdict(lambda: defaultdict(list))


def acquire_rows_in_batch_parallel(k, mk, mask, to_acquire):
    # TODO: This is a version of acquire_new_zf_exp_batch returns mask instead of zf: integrate this nicely
    if mask.size(1) == mk.size(1) == to_acquire.size(1):
        # We are already in a trajectory: every row in to_acquire corresponds to an existing trajectory that
        # we have sampled the next row for.
        m_exp = mask
        mk_exp = mk
    else:
        # We have to initialise trajectories: every row in to_acquire corresponds to the start of a trajectory.
        m_exp = mask.repeat(1, to_acquire.size(1), 1, 1, 1)
        mk_exp = mk.repeat(1, to_acquire.size(1), 1, 1, 1)
    # Loop over slices in batch
    for sl, rows in enumerate(to_acquire):
        # Loop over indices to acquire
        for index, row in enumerate(rows):
            m_exp[sl, index, :, row.item(), :] = 1.
            mk_exp[sl, index, :, row.item(), :] = k[sl, 0, :, row.item(), :]
    return m_exp, mk_exp


# def acquire_row(kspace, masked_kspace, next_rows, mask):
#     zf, mean, std = acquire_new_zf_batch(kspace, masked_kspace, next_rows)
#     # Don't forget to change mask for impro_model (necessary if impro model uses mask)
#     # Also need to change masked kspace for recon model (getting correct next-step zf)
#     # TODO: maybe do this in the acquire_new_zf_batch() function. Doesn't fit with other functions of same
#     #  description, but this one is particularly used for this acquisition loop.
#     for sl, next_row in enumerate(next_rows):
#         mask[sl, :, :, next_row, :] = 1.
#         masked_kspace[sl, :, :, next_row, :] = kspace[sl, :, :, next_row, :]
#     # Get new reconstruction for batch
#     return zf, mean, std, mask, masked_kspace
#
#
# def acquire_row_and_get_new_recon(kspace, masked_kspace, next_rows, mask, recon_model):
#     zf, mean, std, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask)
#     impro_input = create_impro_model_input(args, recon_model, zf, mask)  # TODO: args is global here!
#     return impro_input, zf, mean, std, mask, masked_kspace


def create_data_range_dict(args, loader):
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


def compute_psnr(args, unnorm_recons, gt_exp, data_range):
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


def get_rewards(args, res, mask, masked_kspace, recon_model, gt_mean, gt_std, unnorm_gt,
                data_range, k, comp_psnr=True):
    batch_mk = masked_kspace.view(mask.size(0) * k, 1, res, res, 2)
    # Get new zf: shape = (batch . num_rows x 1 x res x res)
    zf, _, _ = get_new_zf(batch_mk)
    # Get new reconstruction
    impro_input = create_impro_model_input(args, recon_model, zf, mask)
    # shape = batch . k x 1 x res x res, extract reconstruction to compute target
    recons = impro_input[:, 0:1, ...]
    # shape = batch x k x res x res
    recons = recons.view(mask.size(0), k, res, res)
    unnorm_recons = recons * gt_std + gt_mean
    gt_exp = unnorm_gt.expand(-1, k, -1, -1)
    # scores = batch x k (channels), base_score = batch x 1
    ssim_scores = ssim(unnorm_recons, gt_exp, size_average=False, data_range=data_range).mean(-1).mean(-1)
    if comp_psnr:
        psnr_scores = compute_psnr(args, unnorm_recons, gt_exp, data_range)
        return ssim_scores, psnr_scores, impro_input
    return ssim_scores, impro_input


def get_policy_probs(output, unacquired):
    # Reshape policy output such that we can use the same policy for different shapes of acquired
    # This should only be applied in the first acquisition step to initialise trajectories, since after that 'output'
    # should have the shape of batch x num_trajectories x res already.
    if len(output.shape) != len(unacquired.shape):
        output = output.view(output.size(0), 1, output.size(-1))
        output = output.repeat(1, unacquired.size(1), 1)
    # Mask acquired rows
    logits = torch.where(unacquired.byte(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - torch.max(logits, dim=-1, keepdim=True)[0], dim=-1)
    return probs


def train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer, baseline, data_range_dict):
    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)

    res = args.resolution
    k = args.num_trajectories

    # Initialise baseline if it doesn't exist yet (should only happen on epoch 0)
    if args.estimator in ['start_adaptive', 'full_step']:
        if baseline is None:  # only initialise if baseline is not already initialised
            if args.baseline_type == 'step':
                baseline = torch.zeros(args.acquisition_steps)
            elif args.baseline_type == 'statestep':
                # Will become dict of volumes, slice indices, acquisition steps : baseline values
                baseline = {}
            elif args.baseline_type == 'self':
                baseline = None
            elif args.baseline_type == 'selfstep':
                baseline = None
            else:
                raise ValueError(f"{args.baseline.type} is not a valid baseline type.")

    cbatch = 0
    for it, data in enumerate(train_loader):
        # TODO: Use fname, slice to create state-step-dependent baseline
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, sl_idx = data
        cbatch += 1

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

        # Base reconstruction model forward pass to get base score
        impro_input = create_impro_model_input(args, recon_model, zf, mask)
        init_recon = impro_input[:, 0:1, :, :]
        init_unnorm_recon = init_recon * gt_std + gt_mean
        # TODO: Note that this means we train on improvement, rather and absolute score (different from Jin et al.)
        base_score = ssim(init_unnorm_recon, unnorm_gt, size_average=False,
                          data_range=data_range).mean(-1).mean(-1)

        # TODO: implement 3 strategies. Define P = forward pass policy model, M = forward pass reconstruction model,
        #       k = number of trajectories, d = depth of a trajectory (number of acquisition steps).
        #  1) Obtain policy once at the start for each input. Sample trajectories, evaluate reconstruction at the end.
        #     This is adaptive only to the image at the start.
        #     Cost: P + k * M
        #  2) Obtain policy every step using the mask as input. Sample trajectories using this policy at each step.
        #     Evaluate the reconstruction at the end. This is non-adaptive to the image, adaptive to the step.
        #     Alternatively, could condition the policy on the starting image as well. Then it is adaptive to the step
        #     and starting image, but not the current reconstruction at every step.
        #     Cost = k (d P + M)
        #  3) Obtain policy every step using current reconstruction (and mask?) as input. This requires that we compute
        #     The current reconstruction at every step, for all trajectories. This is the full adaptive form.
        #     Cost = k d (P + M)

        if cbatch == 1:
            optimiser.zero_grad()

        # Initialise data tensors
        # Shape = (batch x trajectories x num_steps)
        # action_tensor = torch.zeros((mask.size(0), args.num_trajectories, args.acquisition_steps)).to(args.device)
        # logprob_tensor = torch.zeros((mask.size(0), args.num_trajectories, args.acquisition_steps)).to(args.device)
        # reward_tensor = torch.zeros((mask.size(0), args.num_trajectories, args.acquisition_steps)).to(args.device)
        action_list = []
        logprob_list = []
        reward_list = []

        # Initial policy
        impro_output, _ = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
        # Need to squeeze all dims but batch and row dim here
        unacquired = (mask == 0).squeeze(-4).squeeze(-3).squeeze(-1).float()
        probs = get_policy_probs(impro_output, unacquired)

        for step in range(args.acquisition_steps):
            # Sample initial actions from policy: batch x k
            if step == 0:
                if args.estimator in ['full_local']:  # Wouter's thing
                    # Sampling without replacement
                    actions = torch.multinomial(probs, k, replacement=False)
                elif args.estimator in ['start_adaptive', 'full_step']:  # REINFORCE
                    if args.baseline_type == 'self':  # For now we only support sampling with replacement
                        actions = torch.multinomial(probs, k, replacement=True)
                    else:
                        # Sampling with replacement
                        actions = torch.multinomial(probs, k, replacement=True)
            else:  # Here policy has batch_shape = (batch x num_trajectories), so we just need a sample
                # Since we're only taking a single sample per trajectory, this is 'without replacement'
                policy = torch.distributions.Categorical(probs)
                actions = policy.sample()

            # Store actions and logprobs for later gradient estimation
            # action_tensor[:, :, step] = actions
            action_list.append(actions)

            if step == 0:
                # Parallel sampling of multiple actions
                # probs shape = (batch x res), actions shape = (batch, num_trajectories
                logprobs = torch.log(torch.gather(probs, 1, actions))
            else:
                # Single action per trajectory
                # probs shape = (batch x num_trajectories x res), actions shape = (batch, num_trajectories)
                # Reshape to (batch . num_trajectories x 1\res) for easy gathering
                # Then reshape result back to (batch, num_trajectories)
                selected_probs = torch.gather(
                    probs.view(gt.size(0) * args.num_trajectories, res),
                    1,
                    actions.view(gt.size(0) * args.num_trajectories, 1)).view(actions.shape)
                logprobs = torch.log(selected_probs)

            # logprob_tensor[:, :, step] = logprobs
            if not args.greedy:  # Store logprobs for non-greedy REINFORCE
                logprob_list.append(logprobs)

            # Initial acquisition: add rows to mask in parallel (shape = batch x num_rows x 1\res x res x 2)
            # NOTE: In the first step, this changes mask shape to have size num_rows rather than 1 in dim 1.
            #  This results in unacquired also obtaining this shape. Hence, get_policy_probs requires that
            #  output is also this shape.
            mask, masked_kspace = acquire_rows_in_batch_parallel(kspace, masked_kspace, mask, actions)

            if args.estimator == 'start_adaptive':
                # Not yet at the end: get policy for next acquisition
                if step != args.acquisition_steps - 1:
                    # Mutate unacquired so that we can obtain a new policy on remaining rows
                    # Need to make sure the channel dim remains unsqueezed when k = 1
                    unacquired = (mask == 0).squeeze(-3).squeeze(-1).float()
                    # Get policy on remaining rows (essentially just renormalisation) for next step
                    probs = get_policy_probs(impro_output, unacquired)

                else:  # Final step: only now compute reconstruction and return
                    scores, impro_input = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                      gt_mean, gt_std, unnorm_gt, data_range, k, comp_psnr=False)

                    # Store returns in final index of reward tensor (returns = reward_tensor[:, :, -1])
                    # reward_tensor[:, :, step] = scores - base_score
                    returns = scores - base_score
                    reward_list.append(returns)

                    if args.baseline_type == 'step':
                        # Keep running average step baseline
                        if baseline[step] == 0:  # Need to initialise
                            baseline[step] = returns.mean()
                        baseline[step] = baseline[step] * 0.99 + 0.01 * returns.mean()
                        # Calculate loss
                        # REINFORCE on return
                        loss = -1 * (reward_list[-1] - baseline[-1]) * torch.sum(torch.stack(logprob_list), dim=0)
                    elif args.baseline_type == 'statestep':
                        for i, (vol, sl) in enumerate(zip(fname, sl_idx)):
                            sl = sl.item()
                            # Initialise nested dictionary if not exist
                            if vol not in baseline:
                                baseline[vol] = {}
                            if sl not in baseline[vol]:  # Although we only update the last (step = last step here)
                                baseline[vol][sl] = torch.zeros(args.acquisition_steps)

                            # If just initialised, set baseline to reward
                            if baseline[vol][sl][step] == 0:
                                baseline[vol][sl][step] = returns[i].mean()
                            else:  # Update baseline
                                baseline[vol][sl][step] = baseline[vol][sl][step] * .99 + 0.01 * returns[i].mean()

                        # Calculate loss
                        loss = 0
                        for i, (vol, sl) in enumerate(zip(fname, sl_idx)):
                            sl = sl.item()
                            loss += -1 * ((reward_list[-1] - baseline[vol][sl][-1]) *
                                          torch.sum(torch.stack(logprob_list), dim=0))
                        loss = loss / (i + 1)
                    elif args.baseline_type == 'self':
                        pass
                    elif args.baseline_type == 'selfstep':
                        pass
                    else:
                        raise ValueError(f"{args.baseline.type} is not a valid baseline type.")

                    # Divicde by batches_step to mimic taking mean over larger batch
                    loss = loss.mean() / args.batches_step
                    loss.backward()

                    # Stores 0 loss for all steps but the last
                    epoch_loss[step] += loss.item() / len(train_loader) * gt.size(0) / args.batch_size
                    report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
                    writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

            elif args.estimator in ['full_local', 'full_step']:
                scores, impro_input = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                  gt_mean, gt_std, unnorm_gt, data_range, k, comp_psnr=False)
                # Store rewards shape = (batch x num_trajectories)
                # reward_tensor[:, :, step] = scores - base_score
                reward = scores - base_score
                reward_list.append(reward)

                if args.greedy:  # REINFORCE greedy with step baseline
                    if args.baseline_type == 'step':
                        # Keep running average step baseline
                        if baseline[step] == 0:  # Need to initialise
                            baseline[step] = reward.mean()
                        baseline[step] = baseline[step] * 0.99 + 0.01 * reward.mean()
                        # Use reward within a step
                        loss = -1 * (reward_list[step] - baseline[step]) * logprobs
                    elif args.baseline_type == 'statestep':
                        # Update baseline per state (slice) and per step
                        for i, (vol, sl) in enumerate(zip(fname, sl_idx)):
                            sl = sl.item()
                            # Initialise nested dictionary if not exist
                            if vol not in baseline:
                                baseline[vol] = {}
                            if sl not in baseline[vol]:
                                baseline[vol][sl] = torch.zeros(args.acquisition_steps)

                            # If just initialised, set baseline to reward
                            if baseline[vol][sl][step] == 0:
                                baseline[vol][sl][step] = reward[i].mean()
                            else:  # Update baseline
                                baseline[vol][sl][step] = (baseline[vol][sl][step] * .99 + 0.01 * reward[i].mean())

                        # Calculate loss
                        loss = 0
                        for i, (vol, sl) in enumerate(zip(fname, sl_idx)):
                            sl = sl.item()
                            loss += -1 * (reward_list[step] - baseline[vol][sl][step]) * logprobs
                        loss = loss / (i + 1)
                    elif args.baseline_type in ['self', 'selfstep']:  # identical for greedy
                        # shape = batch x 1
                        avg_reward = torch.mean(reward_list[step], dim=1, keepdim=True)
                        # Get number of trajectories for correct average (see Wouter's paper)
                        num_traj = logprobs.size(-1)
                        # REINFORCE with self-baselines greedy
                        # batch x k
                        loss = -1 * (logprobs * (reward_list[step] - avg_reward)) / (num_traj - 1)
                        # batch
                        loss = loss.sum(dim=1)

                    else:
                        raise ValueError(f"{args.baseline.type} is not a valid baseline type.")

                    # Divicde by batches_step to mimic taking mean over larger batch
                    loss = loss.mean() / args.batches_step
                    loss.backward()

                    epoch_loss[step] += loss.item() / len(train_loader) * gt.size(0) / args.batch_size
                    report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
                    writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

                if args.estimator == 'full_step':
                    if args.baseline_type == 'step':
                        # Keep running average step baseline
                        if baseline[step] == 0:  # Need to initialise
                            baseline[step] = reward.mean()
                        baseline[step] = baseline[step] * 0.99 + 0.01 * reward.mean()
                    elif args.baseline_type == 'statestep':
                        for i, (vol, sl) in enumerate(zip(fname, sl_idx)):
                            sl = sl.item()
                            # Initialise nested dictionary if not exist
                            if vol not in baseline:
                                baseline[vol] = {}
                            if sl not in baseline[vol]:
                                baseline[vol][sl] = torch.zeros(args.acquisition_steps)

                            # If just initialised, set baseline to reward
                            if baseline[vol][sl][step] == 0:
                                baseline[vol][sl][step] = reward[i].mean()
                            else:  # Update baseline
                                baseline[vol][sl][step] = baseline[vol][sl][step] * .99 + 0.01 * reward[i].mean()
                    elif args.baseline_type == 'self':
                        pass
                    elif args.baseline_type == 'selfstep':
                        pass
                    else:
                        raise ValueError(f"{args.baseline.type} is not a valid baseline type.")

                # Set new base_score (we learn improvements)
                base_score = scores

                # If not final step: get policy for next step from current reconstruction
                if step != args.acquisition_steps - 1:
                    # Get policy model output
                    impro_output, _ = impro_model_forward_pass(args, model, impro_input,
                                                               mask.view(gt.size(0) * k, res))
                    # del impro_input
                    # Shape back to batch x num_trajectories x res
                    impro_output = impro_output.view(gt.size(0), k, res)
                    # Need to make sure the channel dim remains unsqueezed when k = 1
                    unacquired = (mask == 0).squeeze(-3).squeeze(-1).float()
                    # Get policy on remaining rows for next step
                    probs = get_policy_probs(impro_output, unacquired)

                # Have all rewards, do REINFORCE / GPOMDP
                else:
                    if args.estimator == 'full_local':
                        # TODO: local gradient estimator (Wouter)
                        raise NotImplementedError
                    elif args.estimator == 'full_step' and not args.greedy:  # GPOMDP with step baseline
                        reward_tensor = torch.stack(reward_list)
                        for step, logprobs in enumerate(logprob_list):
                            # REINFORCE nongreedy with step baseline
                            # GPOMDP: use return from current state onward
                            if args.baseline_type == 'step':
                                loss = -1 * (reward_tensor[step:].sum(dim=0) - baseline[step:].sum()) * logprobs
                            elif args.baseline_type == 'statestep':
                                loss = 0
                                for i, (vol, sl) in enumerate(zip(fname, sl_idx)):
                                    sl = sl.item()
                                    loss += -1 * (reward_tensor[step:].sum(dim=0) -
                                                  baseline[vol][sl][step:].sum()) * logprobs
                                loss = loss / (i + 1)
                            elif args.baseline_type == 'self':
                                # Loss with self-baselines from other trajectories
                                # Only have loss for final step
                                if step != len(logprob_list) - 1:
                                    loss = torch.tensor(0.)
                                else:
                                    # batch x num_trajectories
                                    return_tensor = reward_tensor.sum(dim=0)
                                    # batch x 1
                                    avg_return = torch.mean(return_tensor, dim=1, keepdim=True)
                                    # steps x batch x num_trajectories
                                    logprob_tensor = torch.stack(logprob_list)
                                    # batch x num_trajectories
                                    logprob_tensor_sum = logprob_tensor.sum(0)
                                    # Get number of trajectories for correct average (see Wouter's paper)
                                    num_traj = logprob_tensor.size(-1)
                                    # REINFORCE with self-baselines
                                    # batch x k
                                    loss = -1 * (logprob_tensor_sum * (return_tensor - avg_return)) / (num_traj - 1)
                                    # batch
                                    loss = loss.sum(dim=1)
                                    # Divicde by batches_step to mimic taking mean over larger batch
                                    loss = loss.mean() / args.batches_step
                                    loss.backward()
                            elif args.baseline_type == 'selfstep':
                                gamma_vec = [args.gamma**(t - step) for t in range(step, args.acquisition_steps)]
                                gamma_ten = torch.tensor(gamma_vec).unsqueeze(-1).unsqueeze(-1).to(args.device)
                                # step x batch x 1
                                avg_rewards_tensor = torch.mean(reward_tensor, dim=2, keepdim=True)
                                # Get number of trajectories for correct average (see Wouter's paper)
                                num_traj = logprobs.size(-1)
                                # REINFORCE with self-baselines
                                # batch x k
                                loss = -1 * (logprobs * torch.sum(
                                    gamma_ten * (reward_tensor[step:, :, :] - avg_rewards_tensor[step:, :, :]),
                                    dim=0)) / (num_traj - 1)
                                # batch
                                loss = loss.sum(dim=1)
                            else:
                                raise ValueError(f"{args.baseline.type} is not a valid baseline type.")

                            if args.baseline_type != 'self':
                                # Average over batch (and trajectories except when self baseline)
                                # Divide by batches_step to mimic taking mean over larger batch
                                loss = loss.mean() / args.batches_step
                                loss.backward()  # Store gradients

                            epoch_loss[step] += loss.item() / len(train_loader) * gt.size(0) / args.batch_size
                            report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
                            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)
                    elif args.greedy:  # Complete control flow
                        pass
                    else:
                        raise ValueError('Something went wrong.')
            else:
                raise ValueError(f'{args.estimator} is not a valid estimator!')

        # Backprop if we've reached the prerequisite number of batches
        if cbatch == args.batches_step:
            optimiser.step()
            cbatch = 0

        if it % args.report_interval == 0:
            if it == 0:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, args.report_interval * l * 1e3)
                                      for i, l in enumerate(report_loss)])
            else:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, l * 1e3) for i, l in enumerate(report_loss)])
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}], '
                f'Iter = [{it:4d}/{len(train_loader):4d}], '
                f'Time = {time.perf_counter() - start_iter:.2f}s, '
                f'Avg Loss per step x1e3 = [{loss_str}] ',
            )

            report_loss = [0. for _ in range(args.acquisition_steps)]

        start_iter = time.perf_counter()

    if args.wandb:
        wandb.log({'train_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}}, step=epoch + 1)

    return np.mean(epoch_loss), time.perf_counter() - start_epoch, baseline


def evaluate_recons(args, epoch, recon_model, model, dev_loader, writer, train, data_range_dict):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """
    model.eval()

    epoch_outputs = defaultdict(list)

    ssims, psnrs = 0, 0
    tbs = 0
    start = time.perf_counter()

    res = args.resolution
    k = args.num_dev_trajectories
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, sl_idx = data
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
            impro_input = create_impro_model_input(args, recon_model, zf, mask)

            unnorm_recon = impro_input[:, 0:1, :, :] * gt_std + gt_mean
            init_ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                 data_range=data_range).mean(dim=(-1, -2)).sum()

            # Clamp to min 0 for PSNR computation
            init_psnr_val = psnr(torch.clamp(unnorm_recon, 0., 10.).to('cpu'),
                                 unnorm_gt.to('cpu'),
                                 reduction='none',
                                 data_range=data_range.to('cpu')).sum()

            batch_ssims = [init_ssim_val.item()]
            batch_psnrs = [init_psnr_val.item()]

            # Initial policy
            impro_output, _ = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
            epoch_outputs[0].append(impro_output.to('cpu').numpy())

            # Need to squeeze all dims but batch and row dim here
            unacquired = (mask == 0).squeeze(-4).squeeze(-3).squeeze(-1).float()
            probs = get_policy_probs(impro_output, unacquired)

            for step in range(args.acquisition_steps):
                # Sample initial actions from policy: batch x k
                if step == 0:
                    if args.estimator in ['full_local']:  # Wouter's thing
                        # Sampling without replacement
                        actions = torch.multinomial(probs, k, replacement=False)
                    elif args.estimator in ['start_adaptive', 'full_step']:  # REINFORCE
                        # Sampling with replacement
                        actions = torch.multinomial(probs, k, replacement=True)
                    else:
                        raise ValueError(f"{args.estimator} is not a valid estimator.")
                else:  # Here policy has batch_shape = (batch x num_trajectories), so we just need a sample
                    # Since we're only taking a single sample per trajectory, this is 'without replacement'
                    policy = torch.distributions.Categorical(probs)
                    actions = policy.sample()

                # Initial acquisition: add rows to mask in parallel (shape = batch x num_rows x 1\res x res x 2)
                # NOTE: In the first step, this changes mask shape to have size num_rows rather than 1 in dim 1.
                #  This results in unacquired also obtaining this shape. Hence, get_policy_probs requires that
                #  output is also this shape.
                mask, masked_kspace = acquire_rows_in_batch_parallel(kspace, masked_kspace, mask, actions)

                # Option 1)
                if args.estimator == 'start_adaptive':
                    # Not yet at the end: get policy for next acquisition
                    if step != args.acquisition_steps - 1:
                        # Mutate unacquired so that we can obtain a new policy on remaining rows
                        # Need to make sure the channel dim remains unsqueezed when k = 1
                        unacquired = (mask == 0).squeeze(-3).squeeze(-1).float()
                        # Get policy on remaining rows (essentially just renormalisation) for next step
                        probs = get_policy_probs(impro_output, unacquired)
                        ssim_scores = torch.zeros((gt.size(0), k)) + init_ssim_val / gt.size(0)

                    else:  # Final step: only now compute reconstruction and return
                        ssim_scores, psnr_scores, _ = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                                  gt_mean, gt_std, unnorm_gt, data_range, k)
                        # Predictions are the same every step, so don't need to store anything in epoch_outputs

                # Option 3)
                elif args.estimator in ['full_step', 'full_local']:
                    ssim_scores, psnr_scores, impro_input = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                                        gt_mean, gt_std, unnorm_gt, data_range, k)

                    # If not final step: get policy for next step from current reconstruction
                    if step != args.acquisition_steps - 1:
                        # Get policy model output
                        impro_output, _ = impro_model_forward_pass(args, model, impro_input,
                                                                   mask.view(gt.size(0) * k, res))
                        # del impro_input
                        # Shape back to batch x num_trajectories x res
                        impro_output = impro_output.view(gt.size(0), k, res)
                        # Store predictions, but as mean of trajectories per slice.
                        epoch_outputs[step + 1].append(impro_output.mean(dim=1).to('cpu').numpy())

                        # Mutate unacquired so that we can obtain a new policy on remaining rows
                        # Need to make sure the channel dim remains unsqueezed when k = 1
                        unacquired = (mask == 0).squeeze(-3).squeeze(-1).float()
                        # Get policy on remaining rows (essentially just renormalisation) for next step
                        probs = get_policy_probs(impro_output, unacquired)

                else:
                    raise ValueError(f'{args.estimator} is not a valid estimator!')

                # Average over trajectories, sum over batch dimension
                batch_ssims.append(ssim_scores.mean(dim=1).sum().item())
                batch_psnrs.append(psnr_scores.mean(dim=1).sum().item())

            # shape = al_steps
            ssims += np.array(batch_ssims)
            psnrs += np.array(batch_psnrs)

    ssims /= tbs
    psnrs /= tbs

    if not train:
        # for step in epoch_outputs.keys():
        #     outputs[epoch][step] = np.concatenate(epoch_outputs[step], axis=0).tolist()
        # save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

        for step, val in enumerate(ssims):
            writer.add_scalar('DevSSIM_step{}'.format(step), val, epoch)
            writer.add_scalar('DevPSNR_step{}'.format(step), psnrs[step], epoch)

        if args.wandb:
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
            wandb.log({f'val_ssims.{args.acquisition_steps}': ssims[-1]}, step=epoch + 1)
            wandb.log({f'val_ssims_{args.acquisition_steps}': ssims[-1]}, step=epoch + 1)
            wandb.log({'val_psnrs': {str(key): val for key, val in enumerate(psnrs)}}, step=epoch + 1)
            wandb.log({f'val_psnrs.{args.acquisition_steps}': psnrs[-1]}, step=epoch + 1)
            wandb.log({f'val_psnrs_{args.acquisition_steps}': psnrs[-1]}, step=epoch + 1)
    else:
        if args.wandb:
            wandb.log({'train_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
            wandb.log({'train_psnrs': {str(key): val for key, val in enumerate(psnrs)}}, step=epoch + 1)

    return ssims, psnrs, time.perf_counter() - start


def train_and_eval(args, recon_args, recon_model):
    if args.resume:
        resumed = True
        new_run_dir = args.impro_model_checkpoint.parent
        data_path = args.data_path
        recon_model_checkpoint = args.recon_model_checkpoint

        model, args, start_epoch, optimiser = load_impro_model(pathlib.Path(args.impro_model_checkpoint), optim=True)

        args.old_run_dir = args.run_dir
        args.old_recon_model_checkpoint = args.recon_model_checkpoint
        args.old_data_path = args.data_path

        args.recon_model_checkpoint = recon_model_checkpoint
        args.run_dir = new_run_dir
        args.data_path = data_path
        args.resume = True
    else:
        resumed = False
        model = build_impro_model(args)
        # Add mask parameters for training
        args = add_mask_params(args, recon_args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimiser = build_optim(args, model.parameters())
        start_epoch = 0
        # Create directory to store results in
        savestr = 'res{}_al{}_accel{}_{}_{}_k{}_{}_{}'.format(args.resolution, args.acquisition_steps,
                                                              args.accelerations, args.impro_model_name,
                                                              args.recon_model_name, args.num_trajectories,
                                                              datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                              ''.join(choice(ascii_uppercase) for _ in range(5)))
        args.run_dir = args.exp_dir / savestr
        args.run_dir.mkdir(parents=True, exist_ok=False)

    args.resumed = resumed

    if args.wandb:
        allow_val_change = args.resumed  # only allow changes if resumed: otherwise something is wrong.
        wandb.config.update(args, allow_val_change=allow_val_change)
        wandb.watch(model, log='all')

    # Logging
    logging.info(args)
    logging.info(recon_model)
    logging.info(model)
    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    if not isinstance(model, str):
        # Parameter counting
        logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
            count_parameters(recon_model), count_trainable_parameters(recon_model),
            count_untrainable_parameters(recon_model)))
        logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
            count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)
    elif args.scheduler_type == 'multistep':
        if not isinstance(args.lr_multi_step_size, list):
            args.lr_multi_step_size = [args.lr_multi_step_size]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, args.lr_multi_step_size, args.lr_gamma)
    else:
        raise ValueError("{} is not a valid scheduler choice ('step', 'multistep')".format(args.scheduler_type))

    # Create data loaders
    train_loader, dev_loader, test_loader, display_loader = create_data_loaders(args, shuffle_train=True)
    if args.data_range == 'gt':
        train_data_range_dict = None
        dev_data_range_dict = None
    elif args.data_range == 'volume':
        train_data_range_dict = create_data_range_dict(args, train_loader)
        dev_data_range_dict = create_data_range_dict(args, dev_loader)
    else:
        raise ValueError(f'{args.data_range} is not valid')

    # # TODO: remove this
    # For fully reproducible behaviour: set shuffle_train=False in create_data_loaders
    # train_batch = next(iter(train_loader))
    # train_loader = [train_batch] * 10
    # dev_batch = next(iter(dev_loader))
    # dev_loader = [dev_batch] * 1

    if not args.resume:
        if args.do_train_ssim:
            train_ssims, train_psnrs, train_score_time = evaluate_recons(args, -1, recon_model, model, train_loader,
                                                                         writer, True, train_data_range_dict)
            train_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(train_ssims)])
            train_psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(train_psnrs)])
            logging.info(f'TrainSSIM = [{train_ssims_str}]')
            logging.info(f'TrainPSNR = [{train_psnrs_str}]')
            logging.info(f'TrainScoreTime = {train_score_time:.2f}s')

        dev_ssims, dev_psnrs, dev_score_time = evaluate_recons(args, -1, recon_model, model, dev_loader, writer,
                                                               False, dev_data_range_dict)
        dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(dev_ssims)])
        dev_psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(dev_psnrs)])
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        logging.info(f'  DevPSNR = [{dev_psnrs_str}]')
        logging.info(f'DevScoreTime = {dev_score_time:.2f}s')

    if args.resume and args.baseline_type not in ['self', 'selfstep']:
        raise ValueError('Cannot resume with original baseline for running average baselines.')

    baseline = None

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time, baseline = train_epoch(args, epoch, recon_model, model, train_loader, optimiser,
                                                       writer, baseline, train_data_range_dict)
        dev_loss, dev_loss_time = 0, 0
        dev_ssims, dev_psnrs, dev_score_time = evaluate_recons(args, epoch, recon_model, model, dev_loader, writer,
                                                               False, dev_data_range_dict)

        logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] TrainLoss = {train_loss:.3g} DevLoss = {dev_loss:.3g}'
        )

        if args.do_train_ssim:
            train_ssims, train_psnrs, train_score_time = evaluate_recons(args, epoch, recon_model, model, train_loader,
                                                                         writer, True, train_data_range_dict)
            train_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(train_ssims)])
            train_psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(train_psnrs)])
            logging.info(f'TrainSSIM = [{train_ssims_str}]')
            logging.info(f'TrainPSNR = [{train_psnrs_str}]')
        else:
            train_score_time = 0

        dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(dev_ssims)])
        dev_psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(dev_psnrs)])
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        logging.info(f'  DevPSNR = [{dev_psnrs_str}]')
        logging.info(f'TrainTime = {train_time:.2f}s DevLossTime = {dev_loss_time:.2f}s '
                     f'TrainScoreTime = {train_score_time:.2f}s DevScoreTime = {dev_score_time:.2f}s')

        scheduler.step()

        save_model(args, args.run_dir, epoch, model, optimiser, None, False, args.milestones)

        if args.wandb:
            wandb.save('model.h5')
    writer.close()


def test(args, recon_args, recon_model):
    model, impro_args = load_impro_model(pathlib.Path(args.impro_model_checkpoint))

    impro_args.train_state = args.train_state
    impro_args.dev_state = args.dev_state
    impro_args.test_state = args.test_state

    impro_args.wandb = args.wandb

    if args.wandb:
        wandb.config.update(args)
        wandb.watch(model, log='all')

        # Initialise summary writer
    writer = SummaryWriter(log_dir=impro_args.run_dir / 'summary')

    # Logging
    logging.info(impro_args)
    logging.info(recon_model)
    logging.info(model)

    if not isinstance(model, str):
        # Parameter counting
        logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
            count_parameters(recon_model), count_trainable_parameters(recon_model),
            count_untrainable_parameters(recon_model)))
        logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
            count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))
    # Create data loaders
    _, _, test_loader, _ = create_data_loaders(impro_args, shuffle_train=False)
    if args.data_range == 'gt':
        test_data_range_dict = None
    elif args.data_range == 'volume':
        test_data_range_dict = create_data_range_dict(args, test_loader)
    else:
        raise ValueError(f'{args.data_range} is not valid')

    impro_args.num_dev_trajectories = args.num_dev_trajectories
    test_ssims, test_psnrs, test_score_time = evaluate_recons(impro_args, -1, recon_model, model, test_loader, writer,
                                                              False, test_data_range_dict)
    if args.wandb:
        for epoch in range(args.num_epochs + 1):
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(test_ssims)}}, step=epoch + 1)
            wandb.log({'val_psnrs': {str(key): val for key, val in enumerate(test_psnrs)}}, step=epoch + 1)

    test_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(test_ssims)])
    test_psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(test_psnrs)])
    logging.info(f'  DevSSIM = [{test_ssims_str}]')
    logging.info(f'  DevPSNR = [{test_psnrs_str}]')
    logging.info(f'DevScoreTime = {test_score_time:.2f}s')
    print(test_ssims_str)

    writer.close()


def main(args):
    args.trainQ = True
    args.QR = True
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    check_args_consistency(args, recon_args)

    # Improvement model to train
    if args.do_train:
        train_and_eval(args, recon_args, recon_model)
    else:
        test(args, recon_args, recon_model)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resolution', default=128, type=int, help='Resolution of images')
    parser.add_argument('--dataset', default='fastmri', help='Dataset type to use.')
    parser.add_argument('--challenge', type=str, default='singlecoil',
                        help='Which challenge for fastMRI training.')
    parser.add_argument('--data-path', type=pathlib.Path, default='/var/scratch/tbbakker/data/fastMRI/singlecoil/',
                        help='Path to the dataset. Required for fastMRI training.')
    parser.add_argument('--data-range', type=str, default='volume',
                        help="Type of data range to use. Options are 'volume' and 'gt'.")

    parser.add_argument('--sample-rate', type=float, default=0.04,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str2none, default='CORPD_FBK',
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')
    parser.add_argument('--recon-model-checkpoint', type=pathlib.Path,
                        default='/var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res80_8to4in2_PD_cvol_ch16_b64_wd1em2/model.pt',
                        help='Path to a pretrained reconstruction model. If None then recon-model-name should be'
                        'set to zero_filled.')
    parser.add_argument('--recon-model-name', default='nounc',
                        help='Reconstruction model name corresponding to model checkpoint.')
    parser.add_argument('--impro-model-name', default='convpool',
                        help='Improvement model name (if using resume, must correspond to model at the '
                        'improvement model checkpoint.')
    parser.add_argument('--estimator', default='start_adaptive',
                        help='How to estimate gradients.')
    parser.add_argument('--baseline-type', type=str, default='step',
                        help="Type of baseline to use. Options are 'step', 'statestep', and 'self'.")
    parser.add_argument('--num-trajectories', type=int, default=10, help='Number of trajectories to sample for train.')
    parser.add_argument('--num-dev-trajectories', type=int, default=10, help='Number of trajectories to sample eval.')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='/var/scratch/tbbakker/mrimpro/sweep_results/',
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                        'in provided directory each run')
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
    parser.add_argument('--num-pools', type=int, default=4, help='Number of ConvNet pooling layers. Note that setting '
                        'this too high will cause size mismatch errors, due to even-odd errors in calculation for '
                        'layer size post-flattening.')
    parser.add_argument('--of-which-four-pools', type=int, default=0, help='Number of of the num-pools pooling layers '
                        "that should 4x4 pool instead of 2x2 pool. E.g. if 2, first 2 layers will 4x4 pool, rest will "
                        "2x2 pool. Only used for 'pool' models.")
    parser.add_argument('--drop-prob', type=float, default=0, help='Dropout probability')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Strength of weight decay regularization. TODO: this currently breaks because many weights'
                        'are not updated every step (since we select certain targets only); FIX THIS.')
    parser.add_argument('--pool-stride', default=1, type=int, help='Each how many layers to do max pooling.')

    # Bools
    parser.add_argument('--greedy', type=str2bool, default=False,
                        help="Whether to do greedy REINFORCE (only works with 'full_step' estimator).")
    parser.add_argument('--use-sensitivity', type=str2bool, default=False,
                        help='Whether to use reconstruction model sensitivity as input to the improvement model.')
    parser.add_argument('--center-volume', type=str2bool, default=True,
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--data-parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--do-train-ssim', type=str2bool, default=True,
                        help='Whether to compute SSIM values on training data.')

    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-multi-step-size', nargs='+', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--scheduler-type', type=str, choices=['step', 'multistep'], default='step',
                        help='Number of training epochs')

    # Sweep params
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')

    parser.add_argument('--num-chans', type=int, default=16, help='Number of ConvNet channels')
    parser.add_argument('--in-chans', default=2, type=int, help='Number of image input channels'
                        'E.g. set to 2 if input is reconstruction and uncertainty map')
    parser.add_argument('--fc-size', default=512, type=int, help='Size (width) of fully connected layer(s).')

    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use-data-state', type=str2bool, default=False,
                        help='Whether to use fixed data state for random data selection.')

    parser.add_argument('--batches-step', type=int, default=1,
                        help='Number of batches to compute before doing an optimizer step.')
    parser.add_argument('--do_train', type=str2bool, default=True,
                        help='Number of batches to compute before doing an optimizer step.')
    parser.add_argument('--impro-model-checkpoint', type=pathlib.Path,
                        default='/var/scratch/tbbakker/mrimpro/path/to/model.pt',
                        help='Path to a pretrained impro model.')

    parser.add_argument('--milestones', nargs='+', type=int, default=[0, 9, 19, 29, 39, 49],
                        help='Epochs at which to save model separately.')

    parser.add_argument('--wandb',  type=str2bool, default=False,
                        help='Whether to use wandb logging for this run.')

    parser.add_argument('--resume',  type=str2bool, default=False,
                        help='Continue training previous run?')
    parser.add_argument('--run_id', type=str2none, default=None,
                        help='Wandb run_id to continue training from.')

    parser.add_argument('--test_multi',  type=str2bool, default=False,
                        help='Test multiple models in one script?')
    parser.add_argument('--impro_model_list', nargs='+', type=str, default=[None],
                        help='List of model paths for multi-testing.')

    parser.add_argument('--gamma', type=float, default=1,
                        help='Discount factor in RL. Currently only used for non-greedy training.')
    parser.add_argument('--project',  type=str, default='mrimpro',
                        help='Wandb project name to use.')
    parser.add_argument('--original_setting', type=str2bool, default=True,
                        help='Whether to use original data setting used for knee experiments.')
    parser.add_argument('--low_res', type=str2bool, default=False,
                        help='Whether to use a low res full image, rather than a high res small image when cropping.')

    return parser


def wrap_main(args):
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    if args.use_data_state:
        args.train_state = TRAIN_STATE
        args.dev_state = DEV_STATE
        args.test_state = TEST_STATE
    else:
        args.train_state = None
        args.dev_state = None
        args.test_state = None

    args.use_recon_mask_params = False

    args.milestones = args.milestones + [0, args.num_epochs - 1]

    # TODO: Check if multi-testing works (also see QR_sweep)
    if args.wandb:
        if args.resume:
            assert args.run_id is not None, "run_id must be given if resuming with wandb."
            wandb.init(project=args.project, resume=args.run_id)
            # wandb.restore(run_path=f"mrimpro/{args.run_id}")
        elif args.test_multi:
            wandb.init(project=args.project, reinit=True)
        else:
            wandb.init(project=args.project, config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False

    main(args)


if __name__ == '__main__':
    # To fix known issue with h5py + multiprocessing
    # See: https://discuss.pytorch.org/t/incorrect-data-using-h5py-with-dataloader/7079/2?u=ptrblck
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    base_args = create_arg_parser().parse_args()

    if base_args.test_multi:
        assert not base_args.do_train, "Doing multiple model testing: do_train must be False."
        assert base_args.impro_model_list is not None, "Doing multiple model testing: must have list of impro models."

        for model in base_args.impro_model_list:
            args = copy.deepcopy(base_args)
            args.impro_model_checkpoint = model
            wrap_main(args)
    else:
        wrap_main(base_args)
