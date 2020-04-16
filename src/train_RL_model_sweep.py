"""
Script for running sweeps on DAS5.

Run as follows in mrimpro base dir:

> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages wandb sweep src/RL_sweep.yaml
> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages wandb agent timsey/mrimpro/SWEEP_ID

"""


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

import sys
# sys.path.insert(0, '/home/timsey/Projects/mrimpro/')  # noqa: F401
# sys.path.insert(0, '/Users/tbakker/Projects/mrimpro/')  # noqa: F401
sys.path.insert(0, '/var/scratch/tbbakker/mrimpro/')  # noqa: F401

from src.helpers.torch_metrics import ssim
from src.helpers.utils import add_mask_params, save_json, check_args_consistency
from src.helpers.data_loading import create_data_loaders
from src.recon_models.recon_model_utils import (acquire_new_zf_exp_batch, get_new_zf, acquire_new_zf_batch,
                                                recon_model_forward_pass, create_impro_model_input, load_recon_model)
from src.impro_models.impro_model_utils import build_impro_model, build_optim, save_model, impro_model_forward_pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

targets = defaultdict(lambda: defaultdict(lambda: 0))
target_counts = defaultdict(lambda: defaultdict(lambda: 0))
outputs = defaultdict(lambda: defaultdict(list))


def get_rewards(args, res, mask, masked_kspace, recon_model, gt_mean, gt_std, unnorm_gt, data_range, k):
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
    scores = ssim(unnorm_recons, gt_exp, size_average=False, data_range=data_range).mean(-1).mean(-1)
    return scores, impro_input


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


def train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer, baseline):
    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)

    res = args.resolution
    k = args.num_trajectories

    # Initialise baseline if it doesn't exist yet (should only happen on epoch 0)
    if args.estimator in ['start_adaptive', 'full_step']:
        if baseline is None:
            baseline = torch.zeros(args.acquisition_steps)

    for it, data in enumerate(train_loader):
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, _, _ = data
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
        data_range = unnorm_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

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
                    # Sampling with replacement
                    actions = torch.multinomial(probs, k, replacement=True)
                    # if epoch == 10:
                    #     a = 1
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
                                                      gt_mean, gt_std, unnorm_gt, data_range, k)

                    # Store returns in final index of reward tensor (returns = reward_tensor[:, :, -1])
                    # reward_tensor[:, :, step] = scores - base_score
                    reward_list.append(scores - base_score)

                    # Keep running average step baseline
                    if baseline[step] == 0:  # Need to initialise
                        baseline[step] = (scores - base_score).mean()
                    baseline[step] = baseline[step] * 0.99 + 0.01 * (scores - base_score).mean()

                    # Calculate loss
                    # REINFORCE on return with 0 baseline
                    loss = -1 * (reward_list[-1] - baseline[-1]) * torch.sum(torch.stack(logprob_list), dim=0)
                    loss = loss.mean()  # Average over batch and trajectories
                    loss.backward()

                    # Stores 0 loss for all steps but the last
                    epoch_loss[step] += loss.item() / len(train_loader) * gt.size(0) / args.batch_size
                    report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
                    writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

            elif args.estimator in ['full_local', 'full_step']:
                scores, impro_input = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                  gt_mean, gt_std, unnorm_gt, data_range, k)
                # Store rewards shape = (batch x num_trajectories)
                # reward_tensor[:, :, step] = scores - base_score
                reward_list.append(scores - base_score)

                if args.greedy:  # REINFORCE greedy with step baseline
                    # Use reward within a step
                    loss = -1 * (reward_list[step] - baseline[step]) * logprobs
                    loss = loss.mean()
                    loss.backward()

                    epoch_loss[step] += loss.item() / len(train_loader) * gt.size(0) / args.batch_size
                    report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
                    writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

                if args.estimator == 'full_step':
                    # Keep running average step baseline
                    if baseline[step] == 0:  # Need to initialise
                        baseline[step] = (scores - base_score).mean()
                    baseline[step] = baseline[step] * 0.99 + 0.01 * (scores - base_score).mean()

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
                        pass
                    elif args.estimator == 'full_step' and not args.greedy:  # GPOMDP with step baseline
                        # for step in range(reward_tensor.size(-1)):
                        #     baseline = 0
                        #     # GPOMDP: use return from current state onward
                        #     logprobs = logprob_tensor[:, :, step]
                        #     loss = -1 * (reward_tensor[:, :, step:].sum(dim=-1) - baseline) * logprobs
                        #     loss = loss.mean()  # Average over batch and trajectories
                        #     loss.backward()
                        reward_tensor = torch.stack(reward_list)
                        # for step in reversed(range(args.acquisition_steps)):
                        #     baseline = 0
                        #     # GPOMDP: use return from current state onward
                        #     logprobs = logprob_list.pop(step)
                        #     loss = -1 * (reward_tensor[step:].sum(dim=0) - baseline) * logprobs
                        #     del logprobs
                        #     loss = loss.mean()  # Average over batch and trajectories
                        #     loss.backward()
                        for step, logprobs in enumerate(logprob_list):
                            # REINFORCE nongreedy with step baseline
                            # GPOMDP: use return from current state onward
                            loss = -1 * (reward_tensor[step:].sum(dim=0) - baseline[step:].sum()) * logprobs
                            loss = loss.mean()  # Average over batch and trajectories
                            loss.backward()

                            epoch_loss[step] += loss.item() / len(train_loader) * gt.size(0) / args.batch_size
                            report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
                            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)
                    elif args.greedy:
                        pass
                    else:
                        raise ValueError('Something went wrong.')
            else:
                raise ValueError(f'{args.estimator} is not a valid estimator!')

        optimiser.step()

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


def acquire_row(kspace, masked_kspace, next_rows, mask):
    zf, mean, std = acquire_new_zf_batch(kspace, masked_kspace, next_rows)
    # Don't forget to change mask for impro_model (necessary if impro model uses mask)
    # Also need to change masked kspace for recon model (getting correct next-step zf)
    # TODO: maybe do this in the acquire_new_zf_batch() function. Doesn't fit with other functions of same
    #  description, but this one is particularly used for this acquisition loop.
    for sl, next_row in enumerate(next_rows):
        mask[sl, :, :, next_row, :] = 1.
        masked_kspace[sl, :, :, next_row, :] = kspace[sl, :, :, next_row, :]
    # Get new reconstruction for batch
    return zf, mean, std, mask, masked_kspace


def acquire_row_and_get_new_recon(kspace, masked_kspace, next_rows, mask, recon_model):
    zf, mean, std, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask)
    impro_input = create_impro_model_input(args, recon_model, zf, mask)  # TODO: args is global here!
    return impro_input, zf, mean, std, mask, masked_kspace


def evaluate_recons(args, epoch, recon_model, model, dev_loader, writer, train):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """
    model.eval()

    ssims = 0
    tbs = 0
    start = time.perf_counter()

    res = args.resolution
    k = args.num_dev_trajectories
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, _, _ = data
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
            data_range = unnorm_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

            tbs += mask.size(0)

            # Base reconstruction model forward pass
            impro_input = create_impro_model_input(args, recon_model, zf, mask)

            unnorm_recon = impro_input[:, 0:1, :, :] * gt_std + gt_mean
            init_ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                 data_range=data_range).mean(dim=(-1, -2)).sum()

            batch_ssims = [init_ssim_val.item()]

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
                        # Sampling with replacement
                        actions = torch.multinomial(probs, k, replacement=True)
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
                        scores = torch.zeros((gt.size(0), k)) + init_ssim_val / gt.size(0)

                    else:  # Final step: only now compute reconstruction and return
                        scores, _ = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                gt_mean, gt_std, unnorm_gt, data_range, k)

                # Option 3)
                elif args.estimator in ['full_step', 'full_local']:
                    scores, impro_input = get_rewards(args, res, mask, masked_kspace, recon_model,
                                                      gt_mean, gt_std, unnorm_gt, data_range, k)

                    # If not final step: get policy for next step from current reconstruction
                    if step != args.acquisition_steps - 1:
                        # Get policy model output
                        impro_output, _ = impro_model_forward_pass(args, model, impro_input,
                                                                   mask.view(gt.size(0) * k, res))
                        # del impro_input
                        # Shape back to batch x num_trajectories x res
                        impro_output = impro_output.view(gt.size(0), k, res)
                        # Mutate unacquired so that we can obtain a new policy on remaining rows
                        # Need to make sure the channel dim remains unsqueezed when k = 1
                        unacquired = (mask == 0).squeeze(-3).squeeze(-1).float()
                        # Get policy on remaining rows (essentially just renormalisation) for next step
                        probs = get_policy_probs(impro_output, unacquired)

                else:
                    raise ValueError(f'{args.estimator} is not a valid estimator!')

                # Average over trajectories, sum over batch dimension
                batch_ssims.append(scores.mean(dim=1).sum().item())

            # shape = al_steps
            ssims += np.array(batch_ssims)

    ssims /= tbs

    if not train:
        for step, val in enumerate(ssims):
            writer.add_scalar('DevSSIM_step{}'.format(step), val, epoch)

        if args.wandb:
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
            wandb.log({'val_ssims.10': ssims[-1]}, step=epoch + 1)
            wandb.log({'val_ssims_10': ssims[-1]}, step=epoch + 1)
    else:
        if args.wandb:
            wandb.log({'train_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)

    return ssims, time.perf_counter() - start


def main(args):
    args.trainQ = True
    args.QR = True
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    check_args_consistency(args, recon_args)

    # Improvement model to train
    model = build_impro_model(args)
    # Add mask parameters for training
    args = add_mask_params(args, recon_args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    optimiser = build_optim(args, model.parameters())
    start_epoch = 0
    # Create directory to store results in
    savestr = 'res{}_al{}_accel{}_{}_{}_k{}_{}'.format(args.resolution, args.acquisition_steps, args.accelerations,
                                                       args.impro_model_name, args.recon_model_name,
                                                       args.num_trajectories,
                                                       datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    args.run_dir = args.exp_dir / savestr
    args.run_dir.mkdir(parents=True, exist_ok=False)

    if args.wandb:
        wandb.config.update(args)
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

    # Create data loaders
    train_loader, dev_loader, test_loader, display_loader = create_data_loaders(args, shuffle_train=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)

    # # TODO: remove this
    # For fully reproducible behaviour: set shuffle_train=False in create_data_loaders
    # train_batch = next(iter(train_loader))
    # train_loader = [train_batch] * 10
    # dev_batch = next(iter(dev_loader))
    # dev_loader = [dev_batch] * 1

    if args.do_train_ssim:
        train_ssims, train_ssim_time = evaluate_recons(args, -1, recon_model, model, train_loader, writer, True)
        train_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(train_ssims)])
        logging.info(f'TrainSSIM = [{train_ssims_str}]')
        logging.info(f'TrainSSIMTime = {train_ssim_time:.2f}s')

    dev_ssims, dev_ssim_time = evaluate_recons(args, -1, recon_model, model, dev_loader, writer, False)
    dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(dev_ssims)])
    logging.info(f'  DevSSIM = [{dev_ssims_str}]')
    logging.info(f'DevSSIMTime = {dev_ssim_time:.2f}s')

    baseline = None
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time, baseline = train_epoch(args, epoch, recon_model, model, train_loader,
                                                       optimiser, writer, baseline)
        dev_loss, dev_loss_time = 0, 0
        dev_ssims, dev_ssim_time = evaluate_recons(args, epoch, recon_model, model, dev_loader, writer, False)

        logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] TrainLoss = {train_loss:.3g} DevLoss = {dev_loss:.3g}'
        )

        if args.do_train_ssim:
            train_ssims, train_ssim_time = evaluate_recons(args, epoch, recon_model, model, train_loader, writer, True)
            train_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(train_ssims)])
            logging.info(f'TrainSSIM = [{train_ssims_str}]')
        else:
            train_ssim_time = 0

        dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(dev_ssims)])
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        logging.info(f'TrainTime = {train_time:.2f}s DevLossTime = {dev_loss_time:.2f}s '
                     f'TrainSSIMTime = {train_ssim_time:.2f}s DevSSIMTime = {dev_ssim_time:.2f}s')

        save_model(args, args.run_dir, epoch, model, optimiser, None, False)
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resolution', default=80, type=int, help='Resolution of images')
    parser.add_argument('--dataset', default='fastmri', help='Dataset type to use.')
    parser.add_argument('--challenge', type=str, default='singlecoil',
                        help='Which challenge for fastMRI training.')
    parser.add_argument('--data-path', type=pathlib.Path, default='/var/scratch/tbbakker/data/fastMRI/singlecoil/',
                        help='Path to the dataset. Required for fastMRI training.')
    parser.add_argument('--sample-rate', type=float, default=0.04,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str, default='CORPD_FBK',
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

    # Sweep params
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')

    parser.add_argument('--num-chans', type=int, default=16, help='Number of ConvNet channels')
    parser.add_argument('--in-chans', default=2, type=int, help='Number of image input channels'
                        'E.g. set to 2 if input is reconstruction and uncertainty map')
    parser.add_argument('--fc-size', default=512, type=int, help='Size (width) of fully connected layer(s).')

    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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

    args.use_recon_mask_params = False

    args.wandb = True

    if args.wandb:
        wandb.init(project='mrimpro', config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False

    main(args)