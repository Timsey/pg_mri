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
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot
from tensorboardX import SummaryWriter

from src.helpers.torch_metrics import ssim
from src.helpers.losses import l1_loss_gradfixed, huber_loss
from src.helpers.metrics import Metrics, METRIC_FUNCS
from src.helpers.utils import (add_mask_params, save_json, check_args_consistency, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters, plot_grad_flow)
from src.helpers.data_loading import create_data_loaders
from src.recon_models.recon_model_utils import (acquire_new_zf_exp_batch, acquire_new_zf_batch,
                                                recon_model_forward_pass, create_impro_model_input, load_recon_model)
from src.impro_models.impro_model_utils import (load_impro_model, build_impro_model, build_optim, save_model,
                                                impro_model_forward_pass)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


targets = defaultdict(lambda: defaultdict(lambda: 0))
target_counts = defaultdict(lambda: defaultdict(lambda: 0))
outputs = defaultdict(lambda: defaultdict(list))


def get_pred_and_target(args, kspace, masked_kspace, mask, gt, mean, std, model, recon_model, impro_input, eps, k):
    # Impro model output (Q(a|s) is used to select which targets to actually compute for fine-tuning Q(a|s)
    # squeeze mask such that batch dimension doesn't vanish if it's 1.
    output, train = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
    if not train:  # For center and random model: we don't need to train anything
        return None, output

    recon = impro_input[:, 0:1, ...]  # Other channels are uncertainty maps + other input to the impro model
    norm_recon = recon * std + mean  # Back to original scale for metric  # TODO: is this unnormalisation necessary?

    # shape = batch
    base_score = ssim(norm_recon, gt, size_average=False, data_range=1e-4).mean(-1).mean(-1)  # keep channel dim = 1

    # TODO: efficiently compute forward pass: don't have to do this for rows already acquired. Thus we should
    #  for every batch determine the fraction of rows to train on that have been acquired. Then do the below topk
    #  strategy, ignored already sampled rows. For those topk, we compute next targets using the forward model as
    #  usual (we need to predetermine a fraction for every batch, since the function that acquires target values per
    #  batch requires that we have the same number of rows acquired for every slice). Then we add the specified amount
    #  of zero loss targets (i.e. already acquired rows) randomly.
    res = mask.size(-2)
    batch_acquired_rows = (mask.squeeze(1).squeeze(1).squeeze(-1) == 1)
    acquired_num = batch_acquired_rows[0, :].sum().item()
    # Number of topk rows to use: correct, or sample without replacement instead?
    # A fraction eps of the eventual selected top rows will be randomly chosen. However, for efficiency reasons we
    # must predetermine how many of these randomly chosen rows on average will end up being rows with zero improvement.
    # Eps determines how many of the top rows will be replaced by random, and the fraction of already acquired rows
    # determines how many of these randomly chosen rows will be already acquired.
    # Example: k = 20, res = 80, eps = 0.5, 10 already acquired rows.
    # We choose 20 top rows, of which (after some training) zero will be already acquired.
    # Then we replace a fraction eps (so 10) of these rows by random rows that are not the 10 rows that don't get
    # replaced, so we have 70 options left, of which 10 are already acquired. Thus we expect to see: 10/70 * 10
    # rows with zero improvement. Because we need to determine this beforehand, we determine this number (or rather k
    # minus this number) by doing k Bernoulli trials with probability 65/70, leading to 130/70 * 10 rows with nonzero
    # improvement, of which 10 top rows, and 6/7 * 10 non-top non-zero rows, as expected.
    # I.e. if eps is 0.5 and k = 20, we expect 10 top rows. The other 10 are randomly selected from the remainder, and
    # 10 out of the remaining 70 are zero rows. So 10 top rows, 10/70 * 10 zero rows, and 60/70 * 10 neither.
    # Note that in the algorithm (since we predetermine the amount of topk rows) we sample topk rows to be replaced
    # only after restricting the pool to non-acquired rows. I.e. in the above example we might find 2 zero rows and 18
    # initial top rows. If eps is 0.5, then we expect 10 of those top rows to be replaced, but crucially 2 of the rows
    # that they will be replaced by have already been determined to be zero rows. Thus we really only want to replace
    # 8 of the 18 rows, and thus we need to rescale epsilon to reflect that.
    p = (res - eps * (k + acquired_num)) / (res - eps * k)
    bern = torch.distributions.binomial.Binomial(k, p)
    tk = int(bern.sample())
    # Number of zero rows to use
    zk = k - tk
    # Getting eps * k is equal to getting rescaled_eps * tk + zk in expectation.
    rescaled_eps = (eps * k - zk) / tk

    batch_train_rows = torch.zeros((mask.size(0), tk)).long().to(args.device)
    # Now set random rows to overwrite topk values and inds (epsilon greedily)
    for i in range(mask.size(0)):  # Loop over slices in batch
        # Get k top indices from current output, but make sure to not choose rows that are already acquired here
        acquired_rows = batch_acquired_rows[i, :].nonzero().flatten()
        remaining_row_scores = output[i, :].clone().detach()  # Clone is necessary to prevent gradient explosion
        remaining_row_scores[acquired_rows] = -1e3  # so it doesn't get chosen, but indices stay correct
        _, topk_rows = torch.topk(remaining_row_scores, tk)

        # Some of these will be replaced by random rows: select which ones to replace
        # which of the top k rows to replace with random rows (binary mask)
        topk_replace_mask = (rescaled_eps > torch.rand(tk))
        topk_replace_inds = topk_replace_mask.nonzero().flatten()  # which of the top k rows to replace with random rows
        topk_keep_inds = (topk_replace_mask == 0).nonzero().flatten()
        # Remaining options: no rows that are already acquired
        row_options = torch.tensor([idx for idx in range(res) if
                                    (idx not in acquired_rows and
                                     idx not in topk_rows[topk_keep_inds])]).to(args.device)
        # Shuffle potential rows to acquire random rows (equal to the number of 1s in topk_replace_mask)
        rand_rows = row_options[torch.randperm(row_options.size(0))][:topk_replace_mask.sum()].long()
        # If there are not topk_replace.sum() rows left as options (for instance because we set k == resolution: this
        # would lead to 0 options left) then row_options has length less than topk_replace_inds. This means the below
        # assignment will fail. Hence we clip the length of the latter. We shuffle the topk_replace_inds, so that we
        # don't replace rows the model likes more often than those the model doesn't like (topk_replace_inds is ordered
        # from highest scoring to lowest scoring.
        topk_replace_inds = topk_replace_inds[torch.randperm(topk_replace_inds.size(0))][:len(rand_rows)]

        # TODO: We can just not train already acquired rows, and not sample them during acquisition, since it just
        #  makes the function harder to learn if we do train on them (discontinuous jumps to 0).
        batch_train_rows[i, :] = topk_rows
        batch_train_rows[i, topk_replace_inds] = rand_rows

    # Acquire chosen rows, and compute the improvement target for each (batched)
    # shape = batch x rows = k x res x res
    # TODO: for efficiency, we don't have to do this for acquired rows in principle, since those have target 0
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
    target = output.clone().detach()  # clone is necessary, otherwise changing target also changes output: 0 loss always
    for j, train_rows in enumerate(batch_train_rows):
        # impros[j, 0] (slice j, row 0 in train_rows[j]) corresponds to the row train_rows[j, 0] = 9
        # (for instance). This means the improvement 9th row in the kspace ordering is element 0 in impros.
        kspace_row_inds, permuted_inds = train_rows.sort()
        target[j, kspace_row_inds] = impros[j, permuted_inds]
        # Now randomly set zk of the acquired rows rows to 0
        acquired_rows = batch_acquired_rows[j, :].nonzero().flatten()
        zero_rows = acquired_rows[torch.randperm(acquired_rows.size(0))][:zk]
        target[j, zero_rows] = 0.
    return target, output


def train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer, eps, k):
    # TODO: try batchnorm in FC layers
    if not isinstance(model, str):
        model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)

    epoch_targets = defaultdict(lambda: 0)
    epoch_target_counts = defaultdict(lambda: 0)

    for it, data in enumerate(train_loader):
        kspace, masked_kspace, mask, zf, gt, mean, std, _, _ = data
        # TODO: Maybe normalisation unnecessary for SSIM target?
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
        impro_input = create_impro_model_input(args, recon_model, zf, mask)

        # TODO: number of steps should be dependent on acceleration!
        #  But not all samples in batch have same acceleration: maybe change DataLoader at some point.
        at = time.perf_counter()

        # optimiser.zero_grad()
        for step in range(args.acquisition_steps):
            target, output = get_pred_and_target(args, kspace, masked_kspace, mask, gt, mean, std, model,
                                                 recon_model, impro_input, eps=eps, k=k)

            if target is not None:
                loss_mask = (target != output).float()
                epoch_targets[step + 1] += (target * loss_mask).to('cpu').numpy().sum(axis=0)
                epoch_target_counts[step + 1] += loss_mask.to('cpu').numpy().sum(axis=0)

                # Compute loss and backpropagate
                #  TODO: Maybe use replay buffer type strategy here? Requires too much memory
                #   (need to save all gradients)?
                # loss = l1_loss_gradfixed(output, target, reduction='none')  # TODO: Think about loss function
                loss = huber_loss(output, target, reduction='none')  # TODO: Think about loss function
                loss = loss.mean()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss[step] += loss.item() / len(train_loader)
                report_loss[step] += loss.item() / args.report_interval
                writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

            # Acquire row for next step: GREEDY
            _, next_rows = torch.max(output, dim=1)  # TODO: is greedy a good idea? Acquire multiple maybe?
            impro_input, zf, mean, std, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                          recon_model)

        # optimiser.step()
        if target is not None:
            if args.verbose >= 3:
                logging.info('Time to train single batch of size {} for {} steps: {:.3f}'.format(
                    args.batch_size, args.acquisition_steps, time.perf_counter() - at))

            if it % args.report_interval == 0:
                if it == 0:
                    loss_str = ", ".join(["{}: {:.2f}".format(i + 1, args.report_interval * l * 1e6)
                                          for i, l in enumerate(report_loss)])
                else:
                    loss_str = ", ".join(["{}: {:.2f}".format(i + 1, l * 1e6) for i, l in enumerate(report_loss)])

                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}], '
                    f'Iter = [{it:4d}/{len(train_loader):4d}], '
                    f'Time = {time.perf_counter() - start_iter:.2f}s, '
                    f'Avg Loss per step (x1e6) = [{loss_str}] ',
                )

                report_loss = [0. for _ in range(args.acquisition_steps)]

        start_iter = time.perf_counter()

    if target is not None:
        for step in range(args.acquisition_steps):
            targets[epoch][step + 1] = epoch_targets[step + 1].tolist()
            target_counts[epoch][step + 1] = epoch_target_counts[step + 1].tolist()
        save_json(args.run_dir / 'targets_per_step_per_epoch.json', targets)
        save_json(args.run_dir / 'count_targets_per_step_per_epoch.json', target_counts)

        if args.wandb:
            wandb.log({'train_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}})

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
    impro_input = create_impro_model_input(args, recon_model, zf, mask)
    return impro_input, zf, mean, std, mask, masked_kspace


def evaluate(args, epoch, recon_model, model, dev_loader, writer, k):
    """
    Evaluates using loss function: i.e. how close are the predicted improvements to the actual improvements.
    """
    if not isinstance(model, str):
        model.eval()
    eps = 0  # Only evaluate for acquisitions the model actually wants to do
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    start = time.perf_counter()
    with torch.no_grad():
        for _, data in enumerate(dev_loader):
            kspace, masked_kspace, mask, zf, gt, mean, std, _, _ = data
            # TODO: Maybe normalisation unnecessary for SSIM target?
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
            impro_input = create_impro_model_input(args, recon_model, zf, mask)

            for step in range(args.acquisition_steps):
                # Get impro model input and target:
                # this step requires many forward passes through the reconstruction model
                target, output = get_pred_and_target(args, kspace, masked_kspace, mask, gt, mean, std, model,
                                                     recon_model, impro_input, eps, k)

                if target is not None:
                    # Improvement model output
                    # loss = l1_loss_gradfixed(output, target, reduction='mean')
                    loss = huber_loss(output, target, reduction='mean')  # TODO: Think about loss function
                    epoch_loss[step] += loss.item() / len(dev_loader)  # per batch

                # Greedy policy (size = batch)  # TODO: is this a good idea?
                _, next_rows = torch.max(output, dim=1)
                impro_input, zf, mean, std, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                              recon_model)
        if target is not None:
            if args.wandb:
                wandb.log({'val_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}})
            for step, loss in enumerate(epoch_loss):
                writer.add_scalar('DevLoss_step{}'.format(step), loss, epoch)
    return np.mean(epoch_loss), time.perf_counter() - start


def evaluate_recons(args, epoch, recon_model, model, dev_loader, writer):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """
    if not isinstance(model, str):
        model.eval()

    ssims = 0
    f_ssims = 0
    c_ssims = 0
    r_ssims = 0
    # # strategy: acquisition step: filename: [recons]
    # recons = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # targets = defaultdict(list)
    # metrics = {step: Metrics(METRIC_FUNCS) for step in range(args.acquisition_steps + 1)}
    # c_metrics = {step: Metrics(METRIC_FUNCS) for step in range(args.acquisition_steps + 1)}

    epoch_outputs = defaultdict(list)

    start = time.perf_counter()
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
            kspace, masked_kspace, mask, zf, gt, mean, std, fname, slices = data
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
            impro_input = create_impro_model_input(args, recon_model, zf, mask)

            c_masked_kspace = masked_kspace.clone()
            c_mask = mask.clone()
            f_masked_kspace = masked_kspace.clone()
            f_mask = mask.clone()
            r_masked_kspace = masked_kspace.clone()
            r_mask = mask.clone()
            norm_recon = impro_input[:, 0:1, :, :] * std + mean
            init_ssim_val = ssim(norm_recon, gt, size_average=False, data_range=1e-4).mean()

            batch_ssims = [init_ssim_val.item()]
            f_batch_ssims = [init_ssim_val.item()]
            c_batch_ssims = []
            r_batch_ssims = []
            if epoch == -1:
                c_batch_ssims.append(init_ssim_val.item())
                r_batch_ssims.append(init_ssim_val.item())

            # eval_rec = norm_recon.to('cpu')
            # for i in range(eval_rec.size(0)):
            #     recons[fname[i]]['center'][0].append(eval_rec[i].squeeze().numpy())
            #     recons[fname[i]]['improv'][0].append(eval_rec[i].squeeze().numpy())
            #     targets[fname[i]].append(gt.to('cpu')[i].squeeze().numpy())

            for step in range(args.acquisition_steps):
                # Improvement model output
                output, _ = impro_model_forward_pass(args, model, impro_input, mask.squeeze(1).squeeze(1).squeeze(-1))
                epoch_outputs[step + 1].append(output.to('cpu').numpy())
                # Greedy policy (size = batch)
                # TODO: model often (at least initially) seems to want to choose rows that have already been chosen,
                #  which means it will never choose a new row again for this acquisition trajectory.
                _, next_rows = torch.max(output, dim=1)

                # TODO: necessary? Seems so.
                # Only acquire rows that have not been already acquired
                _, topk_rows = torch.topk(output, args.resolution, dim=1)
                f_unacquired = (f_mask.squeeze(1).squeeze(1).squeeze(-1) == 0)
                f_next_rows = []
                for j, sl_topk_rows in enumerate(topk_rows):
                    for row in sl_topk_rows:
                        if row in f_unacquired[j, :].nonzero().flatten():
                            f_next_rows.append(row)
                            break
                # TODO: moving to device should happen only once before both loops. Should just mutate object after
                f_next_rows = torch.tensor(f_next_rows).long().to(args.device)

                # Acquire this row
                impro_input, zf, mean, std, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                              recon_model)
                norm_recon = impro_input[:, 0:1, :, :] * std + mean
                # shape = 1
                ssim_val = ssim(norm_recon, gt, size_average=False, data_range=1e-4).mean()
                # eventually shape = al_steps
                batch_ssims.append(ssim_val.item())

                # Acquire this row
                f_impro_input, f_zf, f_mean, f_std, f_mask, f_masked_kspace = acquire_row(
                    kspace, f_masked_kspace, f_next_rows, f_mask, recon_model)
                f_norm_recon = f_impro_input[:, 0:1, :, :] * f_std + f_mean
                # shape = 1
                f_ssim_val = ssim(f_norm_recon, gt, size_average=False, data_range=1e-4).mean()
                # eventually shape = al_steps
                f_batch_ssims.append(f_ssim_val.item())

                # eval_rec = norm_recon.to('cpu')
                # for i in range(eval_rec.size(0)):
                #     recons[fname[i]]['improv'][step + 1].append(eval_rec[i].squeeze().numpy())

                # Acquire center row
                if epoch == -1:
                    c_flat_unacq = (c_mask[0].squeeze() == 0).nonzero().flatten()
                    c_next_row = c_flat_unacq[len(c_flat_unacq) // 2]
                    c_next_rows = torch.tensor([c_next_row] * c_mask.size(0)).long().to(args.device)
                    c_impro_input, c_zf, c_mean, c_std, c_mask, c_masked_kspace = acquire_row(kspace, c_masked_kspace,
                                                                                              c_next_rows, c_mask,
                                                                                              recon_model)
                    c_norm_recon = c_impro_input[:, 0:1, :, :] * c_std + c_mean
                    # shape = 1
                    c_ssim_val = ssim(c_norm_recon, gt, size_average=False, data_range=1e-4).mean()
                    # eventually shape = al_steps
                    c_batch_ssims.append(c_ssim_val.item())

                    # c_eval_rec = c_norm_recon.to('cpu')
                    # for i in range(c_eval_rec.size(0)):
                    #     recons[fname[i]]['center'][step + 1].append(c_eval_rec[i].squeeze().numpy())

                    # Acquire random row
                    r_next_rows_options = [(r_mask[i].squeeze() == 0).nonzero().flatten() for i in range(mask.size(0))]
                    r_next_rows = torch.tensor([row_options[int(torch.rand(1) * len(row_options))]
                                                for row_options in r_next_rows_options])
                    r_impro_input, r_zf, r_mean, r_std, r_mask, r_masked_kspace = acquire_row(kspace, r_masked_kspace,
                                                                                              r_next_rows, r_mask,
                                                                                              recon_model)
                    r_norm_recon = r_impro_input[:, 0:1, :, :] * r_std + r_mean
                    # shape = 1
                    r_ssim_val = ssim(r_norm_recon, gt, size_average=False, data_range=1e-4).mean()
                    # eventually shape = al_steps
                    r_batch_ssims.append(r_ssim_val.item())

            # shape = al_steps
            ssims += np.array(batch_ssims) / len(dev_loader)
            f_ssims += np.array(f_batch_ssims) / len(dev_loader)
            c_ssims += np.array(c_batch_ssims) / len(dev_loader)
            r_ssims += np.array(r_batch_ssims) / len(dev_loader)

    # for fname, vol_gts in targets.items():
    #     for step in range(args.acquisition_steps + 1):
    #         if epoch == -1:
    #             c_metrics[step].push(np.stack(vol_gts), np.stack(recons[fname]['center'][step]))
    #         metrics[step].push(np.stack(vol_gts), np.stack(recons[fname]['improv'][step]))
    # print('\n')
    # if epoch == -1:
    #     for step in range(args.acquisition_steps + 1):
    #         print('Center Metrics, step {}: {}'.format(step, c_metrics[step]))
    # print('\n')
    # for step in range(args.acquisition_steps + 1):
    #     print('Improv Metrics, step {}: {}'.format(step, metrics[step]))
    # print('\n')

    for step in range(args.acquisition_steps):
        outputs[epoch][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
    save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

    for step, val in enumerate(ssims):
        writer.add_scalar('DevSSIM_step{}'.format(step), val, epoch)

    if args.wandb:
        wandb.log({'val_ssims': {str(key): val for key, val in enumerate(ssims)}})
        wandb.log({'val_f_ssims': {str(key): val for key, val in enumerate(f_ssims)}})

    return ssims, f_ssims, c_ssims, r_ssims, time.perf_counter() - start


def visualise(args, epoch, model, display_loader, writer):
    # TODO: What to visualise here?
    pass


def main(args):
    args.trainQ = True
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    check_args_consistency(args, recon_args)

    # Improvement model to train
    if args.resume:
        checkpoint, model, optimiser = load_impro_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_impro_model(args)
        # Add mask parameters for training
        args = add_mask_params(args, recon_args)
        if not isinstance(model, str):
            if args.data_parallel:
                model = torch.nn.DataParallel(model)
            optimiser = build_optim(args, model.parameters())
        else:
            optimiser = None
        best_dev_loss = 1e9
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
        if not isinstance(model, str):
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
        if args.verbose >= 1:
            logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
                count_parameters(recon_model), count_trainable_parameters(recon_model),
                count_untrainable_parameters(recon_model)))
            logging.info('Improvement model parameters: total {}, of which {} trainable and {} untrainable'.format(
                count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))
        if args.verbose >= 3:
            for p in model.parameters():
                logging.info(p.shape, p.numel())

    # Create data loaders
    train_loader, dev_loader, test_loader, display_loader = create_data_loaders(args)

    # TODO: remove this
    # first_batch = next(iter(train_loader))
    # train_loader = [first_batch] * 10
    # dev_loader = [first_batch]

    if optimiser is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)
    else:
        scheduler = None

    # Training and evaluation
    k = args.num_target_rows

    dev_ssims, f_dev_ssims, c_dev_ssims, r_dev_ssims, dev_ssim_time = evaluate_recons(
        args, -1, recon_model, model, dev_loader, writer)

    dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(dev_ssims)])
    f_dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(f_dev_ssims)])
    c_dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(c_dev_ssims)])
    r_dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(r_dev_ssims)])
    logging.info(f'C_DevSSIM = [{c_dev_ssims_str}]')
    logging.info(f'R_DevSSIM = [{r_dev_ssims_str}]')
    logging.info(f'F_DevSSIM = [{f_dev_ssims_str}]')
    logging.info(f'  DevSSIM = [{dev_ssims_str}]')
    logging.info(f'DevSSIMTime = {dev_ssim_time:.2f}s')

    for epoch in range(start_epoch, args.num_epochs):
        if scheduler is not None:
            scheduler.step(epoch)
        # eps decays a factor e after num_epochs / eps_decay_rate epochs: i.e. after 1/eps_decay_rate of all epochs
        # This way, the same factor decay happens after the same fraction of epochs, for various values for num_epochs
        # Examples for eps_decay_rate = 5:
        # E.g. if num_epochs = 10, we decay a factor e after 2 epochs: 1/5th of all epochs
        # E.g. if num_epochs = 50, we decay a factor e after 10 epochs: 1/5th of all epochs
        eps = np.exp(np.log(args.start_eps) - args.eps_decay_rate * epoch / args.num_epochs)
        train_loss, train_time = train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer, eps, k)
        # TODO: do both of these? Make more efficient?
        dev_ssims, f_dev_ssims, _, _, dev_ssim_time = evaluate_recons(
            args, epoch, recon_model, model, dev_loader, writer)
        # visualise(args, epoch, model, display_loader, writer)

        if args.do_dev_loss:
            dev_loss, dev_loss_time = evaluate(args, epoch, recon_model, model, dev_loader, writer, k)
            if not isinstance(model, str):
                is_new_best = dev_loss < best_dev_loss
                best_dev_loss = min(best_dev_loss, dev_loss)
                save_model(args, args.run_dir, epoch, model, optimiser, best_dev_loss, is_new_best)
                if args.wandb:
                    wandb.save('model.h5')
        else:
            dev_loss = 0
            dev_loss_time = 0

        dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(dev_ssims)])
        f_dev_ssims_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(f_dev_ssims)])
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.3g} '
            f'DevLoss = {dev_loss:.3g} TrainTime = {train_time:.2f}s '
            f'DevLossTime = {dev_loss_time:.2f}s DevSSIMTime = {dev_ssim_time:.2f}s',
        )
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        logging.info(f'F_DevSSIM = [{f_dev_ssims_str}]')
        # save_model(args, args.run_dir, epoch, model, optimiser, None, False)
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--dataset', choices=['fastmri', 'cifar10'], required=True,
                        help='Dataset to use.')
    parser.add_argument('--wandb',  action='store_true',
                        help='Whether to use wandb logging for this run.')

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
    parser.add_argument('--use-sensitivity',  action='store_true',
                        help='Whether to use reconstruction model sensitivity as input to the improvement model.')
    parser.add_argument('--num-sens-samples', type=int, default=10,
                        help='Number of reconstruction model samples to average the sensitivity map over.')

    parser.add_argument('--center-volume', action='store_true',
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--use-recon-mask-params', action='store_true',
                        help='Whether to use mask parameter settings (acceleration and center fraction) that the '
                        'reconstruction model was trained on. This will overwrite any other mask settings.')

    parser.add_argument('--impro-model-name', choices=['convpool', 'convpoolmask', 'convbottle', 'maskfc', 'maskconv',
                                                       'convpoolmaskconv', 'location', 'center', 'random'],
                        required=True, help='Improvement model name (if using resume, must correspond to model at the '
                        'improvement model checkpoint.')
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
    parser.add_argument('--num-pools', type=int, default=4, help='Number of ConvNet pooling layers. Note that setting '
                        'this too high will cause size mismatch errors, due to even-odd errors in calculation for '
                        'layer size post-flattening.')
    parser.add_argument('--of-which-four-pools', type=int, default=2, help='Number of of the num-pools pooling layers '
                        "that should 4x4 pool instead of 2x2 pool. E.g. if 2, first 2 layers will 4x4 pool, rest will "
                        "2x2 pool. Only used for 'pool' models.")
    parser.add_argument('--drop-prob', type=float, default=0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of ConvNet channels')
    parser.add_argument('--in-chans', default=2, type=int, help='Number of image input channels'
                        'E.g. set to 2 if input is reconstruction and uncertainty map')
    parser.add_argument('--out-chans', type=int, default=32, help='Number of ConvNet output channels: these are input '
                        "for the FC layers that follow. Only used for 'bottle' models.")
    parser.add_argument('--fc-size', default=512, type=int, help='Size (width) of fully connected layer(s).')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Strength of weight decay regularization. TODO: this currently breaks because many weights'
                        'are not updated every step (since we select certain targets only); FIX THIS.')

    parser.add_argument('--start-eps', type=float, default=0.5, help='Epsilon to start with. This determines '
                        'the trade-off between training on rows the improvement networks suggests, and training on '
                        'randomly sampled rows. Note that rows are selected mostly randomly at the start of training '
                        'regardless, since the initialised model will not have learned anything useful yet.')
    parser.add_argument('--eps-decay-rate', type=float, default=5, help='Epsilon decay rate. Epsilon decays a '
                        'factor e after a fraction 1/eps_decay_rate of all epochs have passed.')
    parser.add_argument('--num-target-rows', type=int, default=10, help='Number of rows to compute ground truth '
                        'targets for every update step.')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, required=True,
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                        'in provided directory each run')

    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    parser.add_argument('--do-dev-loss', action='store_true',
                        help='Whether to compute dev loss during training (generally takes ~1/5th of train time.'
                             'Not to be confused with SSIM evaluation, which is always done.')
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
