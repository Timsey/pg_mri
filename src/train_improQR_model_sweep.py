"""
Script for running sweeps on DAS5.

Run as follows in mrimpro base dir:

> CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages wandb sweep src/QR_sweep.yaml
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

from src.helpers.gumbel import compute_log_R
from src.helpers.torch_metrics import ssim
from src.helpers.utils import (add_mask_params, save_json, check_args_consistency, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters, str2bool, str2none)
from src.helpers.data_loading import create_data_loaders
from src.helpers.states import DEV_STATE, TRAIN_STATE, TEST_STATE
from src.recon_models.recon_model_utils import (acquire_new_zf_exp_batch, acquire_new_zf_batch,
                                                recon_model_forward_pass, create_impro_model_input, load_recon_model)
from src.impro_models.impro_model_utils import (build_impro_model, load_impro_model, build_optim, save_model,
                                                impro_model_forward_pass)
from src.helpers import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

targets = defaultdict(lambda: defaultdict(lambda: 0))
target_counts = defaultdict(lambda: defaultdict(lambda: 0))
outputs = defaultdict(lambda: defaultdict(list))


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


def reinforce_unordered(cost, log_p, baseline=True):
    with torch.no_grad():  # Don't compute gradients for advantage and ratio
        log_R_s, log_R_ss = compute_log_R(log_p)
        bl_vals = ((log_p[:, None, :] + log_R_ss).exp() * cost[:, None, :]).sum(-1)
        adv_B = cost - bl_vals if baseline else cost
    return ((log_p + log_R_s).exp() * adv_B).sum(-1)


def get_rewards(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model, impro_input,
                actions, output, data_range):
    # actions is a batch x k tensor, containing row indices to compute targets for

    recon = impro_input[:, 0:1, ...]  # Other channels are uncertainty maps + other input to the impro model
    unnorm_recon = recon * gt_std + gt_mean  # Back to original scale for metric

    # shape = batch x 1
    base_score = ssim(unnorm_recon, unnorm_gt, size_average=False,
                      data_range=data_range).mean(-1).mean(-1)  # keep channel dim = 1

    res = mask.size(-2)
    # Acquire chosen rows, and compute the improvement target for each (batched)
    # shape = batch x rows = k x res x res
    zf_exp, _, _ = acquire_new_zf_exp_batch(kspace, masked_kspace, actions)
    # shape = batch . k x 1 x res x res, so that we can run the forward model for all rows in the batch
    zf_input = zf_exp.view(actions.size(0) * actions.size(1), 1, res, res)
    # shape = batch . k x 2 x res x res
    recons_output = recon_model_forward_pass(args, recon_model, zf_input)
    # shape = batch . k x 1 x res x res, extract reconstruction to compute target
    recons = recons_output[:, 0:1, ...]
    # shape = batch x k x res x res
    recons = recons.view(actions.size(0), actions.size(1), res, res)
    unnorm_recons = recons * gt_std + gt_mean  # TODO: Normalisation necessary?
    gt_exp = unnorm_gt.expand(-1, actions.size(1), -1, -1)
    # scores = batch x k (channels), base_score = batch x 1
    scores = ssim(unnorm_recons, gt_exp, size_average=False, data_range=data_range).mean(-1).mean(-1)
    impros = (scores - base_score) * 1  # TODO: is this 'normalisation'?
    # target = batch x rows, batch_train_rows and impros = batch x k
    # target = torch.zeros(actions.size(0), res).to(args.device)
    target = output.detach().clone()
    for j, train_rows in enumerate(actions):
        # impros[j, 0] (slice j, row 0 in train_rows[j]) corresponds to the row train_rows[j, 0] = 9
        # (for instance). This means the improvement 9th row in the kspace ordering is element 0 in impros.
        kspace_row_inds, permuted_inds = train_rows.sort()
        target[j, kspace_row_inds] = impros[j, permuted_inds]
    return target


def train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer, k, data_range_dict):
    # TODO: try batchnorm in FC layers
    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)

    epoch_targets = defaultdict(lambda: 0)
    epoch_target_counts = defaultdict(lambda: 0)

    for it, data in enumerate(train_loader):
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
        impro_input = create_impro_model_input(args, recon_model, zf, mask)

        optimiser.zero_grad()
        for step in range(args.acquisition_steps):
            # TODO: Output nans?
            output, _ = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
            loss_mask = (mask == 0).squeeze().float()

            # Mask acquired rows
            logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
            # logits = output

            # Softmax over 'logits' representing row scores
            probs = torch.nn.functional.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
            # # TODO: this possibly samples non-allowed rows sometimes, which have prob ~ 0, and thus log prob -inf.
            # #  To fix this we'd need to restrict the categorical to only allowed rows, keeping track of the indices,
            # #  so that we correctly backpropagate the loss to the model.

            # Also need this for sampling the next row at the end of this loop
            policy = torch.distributions.Categorical(probs)
            if args.estimator == 'wr':
                # batch x k
                actions = policy.sample((k,)).transpose(0, 1)  # TODO: DiCE estimator; differentiable sampling?
            elif args.estimator == 'wor':
                actions = torch.multinomial(probs, k, replacement=False)
            else:
                raise ValueError(f'{args.estimator} is not a valid estimator.')

            # REINFORCE-like with baselines
            target = get_rewards(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                 impro_input, actions, output, data_range)
            epoch_targets[step + 1] += (target * loss_mask).to('cpu').numpy().sum(axis=0)
            epoch_target_counts[step + 1] += loss_mask.to('cpu').numpy().sum(axis=0)

            # TODO: Only works if all actions are unique within a sample
            # batch x k
            action_logprobs = torch.log(torch.gather(probs, 1, actions))
            action_rewards = torch.gather(target, 1, actions)

            if args.estimator == 'wr':
                # With replacement
                # batch x 1
                avg_reward = torch.mean(action_rewards, dim=1, keepdim=True)
                # REINFORCE with self-baselines
                # batch x k
                loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(1) - 1)
                # batch
                loss = loss.sum(dim=1)

            elif args.estimator == 'wor':
                # Without replacement
                # batch x 1
                loss = reinforce_unordered(-1 * action_rewards, action_logprobs)
            else:
                raise ValueError(f'{args.estimator} is not a valid estimator.')

            loss = loss.mean()
            loss.backward()

            epoch_loss[step] += loss.item() / len(train_loader) * mask.size(0) / args.batch_size
            report_loss[step] += loss.item() / args.report_interval * mask.size(0) / args.batch_size
            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

            if args.acq_strat == 'max':
                # Acquire row for next step: GREEDY
                _, next_rows = torch.max(logits, dim=1)  # TODO: is greedy a good idea? Acquire multiple maybe?
            elif args.acq_strat == 'sample':
                # Acquire next row by sampling
                next_rows = policy.sample()
            else:
                raise ValueError(f'{args.acq_strat} is not a valid acquisition strategy')

            impro_input, zf, _, _, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                     recon_model)

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

    # for step in range(args.acquisition_steps):
    #     targets[epoch][step + 1] = epoch_targets[step + 1].tolist()
    #     target_counts[epoch][step + 1] = epoch_target_counts[step + 1].tolist()
    # save_json(args.run_dir / 'targets_per_step_per_epoch.json', targets)
    # save_json(args.run_dir / 'count_targets_per_step_per_epoch.json', target_counts)

    if args.wandb:
        wandb.log({'train_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}}, step=epoch + 1)

    return np.mean(epoch_loss), time.perf_counter() - start_epoch


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
    impro_input = create_impro_model_input(args, recon_model, zf, mask)  # TODO: args is global here!
    return impro_input, zf, mean, std, mask, masked_kspace


def evaluate(args, epoch, recon_model, model, dev_loader, writer, k, data_range_dict):
    # TODO: sorter
    """
    Evaluates using loss function: i.e. how close are the predicted improvements to the actual improvements.
    """
    model.eval()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    start = time.perf_counter()
    global_step = epoch * len(dev_loader)

    with torch.no_grad():
        for it, data in enumerate(dev_loader):
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
            impro_input = create_impro_model_input(args, recon_model, zf, mask)

            for step in range(args.acquisition_steps):
                output, _ = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
                loss_mask = (mask == 0).squeeze().float()

                # Mask acquired rows
                logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
                # logits = output

                # Softmax over 'logits' representing row scores
                probs = torch.nn.functional.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
                # TODO: this possibly samples non-allowed rows sometimes, which have prob ~ 0, and thus log prob -inf.
                #  To fix this we'd need to restrict the categorical to only allowed rows, keeping track of the indices,
                #  so that we correctly backpropagate the loss to the model.
                # Also need this for sampling the next row at the end of this loop
                policy = torch.distributions.Categorical(probs)
                if args.estimator == 'wr':
                    # batch x k
                    actions = policy.sample((k,)).transpose(0, 1)  # TODO: DiCE estimator; differentiable sampling?
                elif args.estimator == 'wor':
                    actions = torch.multinomial(probs, k, replacement=False)
                else:
                    raise ValueError(f'{args.estimator} is not a valid estimator.')

                # REINFORCE-like with baselines
                target = get_rewards(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                     impro_input, actions, output, data_range)

                # TODO: Only works if all actions are unique within a sample
                # batch x k
                action_logprobs = torch.log(torch.gather(probs, 1, actions))
                action_rewards = torch.gather(target, 1, actions)
                # batch x 1
                avg_reward = torch.mean(action_rewards, dim=1, keepdim=True)
                # REINFORCE with self-baselines
                # batch x k
                loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(1) - 1)
                # batch x 1
                loss = loss.sum(dim=1)
                # 1 x 1
                loss = loss.mean()
                epoch_loss[step] += loss.item() / len(dev_loader) * mask.size(0) / args.batch_size

                writer.add_scalar('DevLoss_step{}'.format(step), loss.item(), global_step + it)

                if args.acq_strat == 'max':
                    # Acquire row for next step: GREEDY
                    _, next_rows = torch.max(logits, dim=1)  # TODO: is greedy a good idea? Acquire multiple maybe?
                elif args.acq_strat == 'sample':
                    # Acquire next row by sampling
                    next_rows = policy.sample()
                else:
                    raise ValueError(f'{args.acq_strat} is not a valid acquisition strategy')

                impro_input, zf, _, _, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                         recon_model)

        if args.wandb:
            wandb.log({'dev_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}}, step=epoch + 1)
        for step, loss in enumerate(epoch_loss):
            writer.add_scalar('DevLoss_step{}'.format(step), loss, epoch)
    return np.mean(epoch_loss), time.perf_counter() - start


def evaluate_recons(args, epoch, recon_model, model, dev_loader, writer, train, data_range_dict):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """
    model.eval()

    ssims = 0
    # # strategy: acquisition step: filename: [recons]
    # recons = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # targets = defaultdict(list)

    epoch_outputs = defaultdict(list)
    tbs = 0
    start = time.perf_counter()
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
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
            impro_input = create_impro_model_input(args, recon_model, zf, mask)

            unnorm_recon = impro_input[:, 0:1, :, :] * gt_std + gt_mean
            init_ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                 data_range=data_range).mean(dim=(-1, -2)).sum()

            base_mask = mask
            base_masked_kspace = masked_kspace
            base_impro_input = impro_input

            if args.acq_strat == 'max' and args.num_test_trajectories != 1:
                print('Doing max acquisition strategy: set num_test_trajectories to 1.')
                args.num_test_trajectories = 1
            for _ in range(args.num_test_trajectories):
                mask = base_mask.clone()
                masked_kspace = base_masked_kspace.clone()
                impro_input = base_impro_input.clone()
                batch_ssims = [init_ssim_val.item()]

                for step in range(args.acquisition_steps):
                    # Improvement model output
                    output, _ = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
                    epoch_outputs[step + 1].append(output.to('cpu').numpy())

                    if args.acq_strat == 'max':
                        # Greedy policy (size = batch)
                        # Only acquire rows that have not been already acquired
                        # TODO: Could just take the max of the masked output
                        _, topk_rows = torch.topk(output, args.resolution, dim=1)
                        unacquired = (mask.squeeze(1).squeeze(1).squeeze(-1) == 0)
                        next_rows = []
                        for j, sl_topk_rows in enumerate(topk_rows):
                            for row in sl_topk_rows:
                                if row in unacquired[j, :].nonzero().flatten():
                                    next_rows.append(row)
                                    break
                        # TODO: moving to device should happen only once before both loops. Should just mutate object after
                        next_rows = torch.tensor(next_rows).long().to(args.device)
                    elif args.acq_strat == 'sample':
                        # Acquire rows by sampling
                        loss_mask = (mask == 0).squeeze().float()
                        # Mask acquired rows
                        logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
                        # Softmax over 'logits' representing row scores
                        probs = torch.nn.functional.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
                        policy = torch.distributions.Categorical(probs)
                        # batch x k
                        next_rows = policy.sample()
                    else:
                        raise ValueError(f'{args.acq_strat} is not a valid acquisition strategy')

                    # Acquire this row
                    impro_input, _, _, _, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                            recon_model)

                    # TODO: this is weird. The recon model is trained to reconstruct targets normalised by the original
                    #  zf image's normalisation constants, but now we're un-normalising the outputted reconstruction by
                    #  the few-step-ahead zf image's normalisation constants. This un-normalised output reconstruction
                    #  may thus have a different scale that the target image (which is normalised by the original zf image's
                    #  normalisation constants). We must either use the original zf's constants for un-normalisation when
                    #  computing SSIM scores, or we use the target's normalisation constants, which seems more justified.
                    #  Note that currently we don't do normalisation based on the targets, so this will require some
                    #  changes to the DataLoader.
                    unnorm_recon = impro_input[:, 0:1, :, :] * gt_std + gt_mean
                    # shape = 1
                    ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                    data_range=data_range).mean(dim=(-1, -2)).sum()
                    # eventually shape = al_steps
                    batch_ssims.append(ssim_val.item())

                # shape = al_steps
                ssims += np.array(batch_ssims)

    ssims /= (tbs * args.num_test_trajectories)

    if not train:
        # for step in range(args.acquisition_steps):
        #     outputs[epoch][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
        # save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

        for step, val in enumerate(ssims):
            writer.add_scalar('DevSSIM_step{}'.format(step), val, epoch)

        if args.wandb:
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
            wandb.log({f'val_ssims.{args.acquisition_steps}': ssims[-1]}, step=epoch + 1)
            wandb.log({f'val_ssims_{args.acquisition_steps}': ssims[-1]}, step=epoch + 1)
    else:
        if args.wandb:
            wandb.log({'train_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)

    return ssims, time.perf_counter() - start


def train_and_eval(args, recon_args, recon_model):
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
                                                       args.num_target_rows,
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

    # Training and evaluation
    k = args.num_target_rows

    # TODO: remove this
    # For fully reproducible behaviour: set shuffle_train=False in create_data_loaders
    # train_batch = next(iter(train_loader))
    # train_loader = [train_batch] * 10
    # dev_batch = next(iter(dev_loader))
    # dev_loader = [dev_batch] * 1

    if args.do_train_ssim:
        train_ssims, train_ssim_time = evaluate_recons(args, -1, recon_model, model, train_loader, writer,
                                                       True, train_data_range_dict)
        train_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(train_ssims)])
        logging.info(f'TrainSSIM = [{train_ssims_str}]')
        logging.info(f'TrainSSIMTime = {train_ssim_time:.2f}s')

    dev_ssims, dev_ssim_time = evaluate_recons(args, -1, recon_model, model, dev_loader, writer,
                                               False, dev_data_range_dict)
    dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(dev_ssims)])
    logging.info(f'  DevSSIM = [{dev_ssims_str}]')
    logging.info(f'DevSSIMTime = {dev_ssim_time:.2f}s')

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step()
        train_loss, train_time = train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer,
                                             k, train_data_range_dict)
        # dev_loss, dev_loss_time = evaluate(args, epoch, recon_model, model, dev_loader, writer, k, dev_data_range_dict)
        dev_loss, dev_loss_time = 0, 0
        dev_ssims, dev_ssim_time = evaluate_recons(args, epoch, recon_model, model, dev_loader, writer,
                                                   False, dev_data_range_dict)

        logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] TrainLoss = {train_loss:.3g} DevLoss = {dev_loss:.3g}'
        )

        if args.do_train_ssim:
            train_ssims, train_ssim_time = evaluate_recons(args, epoch, recon_model, model, train_loader, writer,
                                                           True, train_data_range_dict)
            train_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(train_ssims)])
            logging.info(f'TrainSSIM = [{train_ssims_str}]')
        else:
            train_ssim_time = 0

        dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(dev_ssims)])
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        logging.info(f'TrainTime = {train_time:.2f}s DevLossTime = {dev_loss_time:.2f}s '
                     f'TrainSSIMTime = {train_ssim_time:.2f}s DevSSIMTime = {dev_ssim_time:.2f}s')

        save_model(args, args.run_dir, epoch, model, optimiser, None, False, args.milestones)
    writer.close()


def test(args, recon_args, recon_model):
    model, impro_args = load_impro_model(pathlib.Path(args.impro_model_checkpoint))

    impro_args.train_state = args.train_state
    impro_args.dev_state = args.dev_state
    impro_args.test_state = args.test_state

    impro_args.wandb = args.wandb
    impro_args.num_test_trajectories = args.num_test_trajectories

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

    test_ssims, test_ssim_time = evaluate_recons(impro_args, -1, recon_model, model, test_loader, writer,
                                                 False, test_data_range_dict)
    if args.wandb:
        for epoch in range(args.num_epochs):
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(test_ssims)}}, step=epoch + 1)

    test_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(test_ssims)])
    logging.info(f'  DevSSIM = [{test_ssims_str}]')
    logging.info(f'DevSSIMTime = {test_ssim_time:.2f}s')
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
    parser.add_argument('--num-target-rows', type=int, default=10, help='Number of rows to compute ground truth '
                        'targets for every update step.')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')

    parser.add_argument('--estimator', type=str, default='wr',
                        help='Which estimator to use: with replacement (wr), or without replacement (wor)')
    parser.add_argument('--acq_strat', type=str, default='max',
                        help='Which acquisition strategy to use: greedy or sample')

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
    parser.add_argument('--in-chans', default=1, type=int, help='Number of image input channels'
                        'E.g. set to 2 if input is reconstruction and uncertainty map')
    parser.add_argument('--fc-size', default=256, type=int, help='Size (width) of fully connected layer(s).')

    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--scheduler-type', type=str, choices=['step', 'multistep'], default='step',
                        help='Number of training epochs')
    parser.add_argument('--lr-multi-step-size', nargs='+', type=int, default=40,
                        help='Period of learning rate decay')

    parser.add_argument('--use-data-state', type=str2bool, default=False,
                        help='Whether to use fixed data state for random data selection.')

    parser.add_argument('--maskconv-depth', type=int, default=3, help='Number of layers in maskconv model.')

    parser.add_argument('--do_train', type=str2bool, default=True,
                        help='Number of batches to compute before doing an optimizer step.')
    parser.add_argument('--impro-model-checkpoint', type=pathlib.Path,
                        default='/var/scratch/tbbakker/mrimpro/path/to/model.pt',
                        help='Path to a pretrained impro model.')

    parser.add_argument('--milestones', nargs='+', type=int, default=[0, 9, 19, 29, 39, 49],
                        help='Epochs at which to save model separately.')

    parser.add_argument('--wandb',  type=str2bool, default=False,
                        help='Whether to use wandb logging for this run.')
    parser.add_argument('--num-test-trajectories', type=int, default=4,
                        help='Number of trajectories to use when testing sampling policy.')
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

    args.use_recon_mask_params = False

    if args.use_data_state:
        args.train_state = TRAIN_STATE
        args.dev_state = DEV_STATE
        args.test_state = TEST_STATE
    else:
        args.train_state = None
        args.dev_state = None
        args.test_state = None

    args.milestones = args.milestones + [0, args.num_epochs - 1]

    if args.wandb:
        wandb.init(project='mrimpro', config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False

    main(args)
