import torch
import random

from .policy_model_def import build_policy_model
from src.helpers import transforms
from src.helpers.utils import build_optim
from src.helpers.torch_metrics import compute_ssim, compute_psnr


def save_policy_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if epoch in args.milestones:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=exp_dir / f'model_{epoch}.pt'
        )


def load_policy_model(checkpoint_file, optim=False):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_policy_model(args)

    if not optim:
        # No gradients for this model
        for param in model.parameters():
            param.requires_grad = False

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_optim(args, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        return model, args, start_epoch, optimizer

    del checkpoint
    return model, args


def get_new_zf(masked_kspace_batch):
    # Inverse Fourier Transform to get zero filled solution
    image_batch = transforms.ifft2(masked_kspace_batch)
    # Absolute value
    image_batch = transforms.complex_abs(image_batch)
    # Normalize input
    image_batch, means, stds = transforms.normalize(image_batch, dim=(-2, -1), eps=1e-11)
    image_batch = image_batch.clamp(-6, 6)
    return image_batch, means, stds


def acquire_rows_in_batch_parallel(k, mk, mask, to_acquire):
    if mask.size(1) == mk.size(1) == to_acquire.size(1):
        # Two cases:
        # 1) We are only requesting a single k-space column to acquire per batch.
        # 2) We are requesting multiple k-space columns per batch, and we are already in a trajectory of the non-greedy
        # model: every column in to_acquire corresponds to an existing trajectory that we have sampled the next
        # column for.
        m_exp = mask
        mk_exp = mk
    else:
        # We have to initialise trajectories: every row in to_acquire corresponds to a trajectory.
        m_exp = mask.repeat(1, to_acquire.size(1), 1, 1, 1)
        mk_exp = mk.repeat(1, to_acquire.size(1), 1, 1, 1)
    # Loop over slices in batch
    for sl, rows in enumerate(to_acquire):
        # Loop over indices to acquire
        for index, row in enumerate(rows):  # Will only be single index if first case (see comment above)
            m_exp[sl, index, :, row.item(), :] = 1.
            mk_exp[sl, index, :, row.item(), :] = k[sl, 0, :, row.item(), :]
    return m_exp, mk_exp


def compute_next_step_reconstruction(recon_model, kspace, masked_kspace, mask, next_rows):
    # This computation is done by reshaping the masked k-space tensor to (batch . num_trajectories x 1 x res x res)
    # and then reshaping back after performing a reconstruction.
    mask, masked_kspace = acquire_rows_in_batch_parallel(kspace, masked_kspace, mask, next_rows)
    channel_size = masked_kspace.shape[1]
    res = masked_kspace.size(-2)
    # Combine batch and channel dimension for parallel computation if necessary
    masked_kspace = masked_kspace.view(mask.size(0) * channel_size, 1, res, res, 2)
    zf, _, _ = get_new_zf(masked_kspace)
    recon = recon_model(zf)

    # Reshape back to B X C (=parallel acquisitions) x H x W
    recon = recon.view(mask.size(0), channel_size, res, res)
    zf = zf.view(mask.size(0), channel_size, res, res)
    masked_kspace = masked_kspace.view(mask.size(0), channel_size, res, res, 2)
    return mask, masked_kspace, zf, recon


def get_policy_probs(model, recons, mask):
    channel_size = mask.shape[1]
    res = mask.size(-2)
    # Reshape trajectory dimension into batch dimension for parallel forward pass
    recons = recons.view(mask.size(0) * channel_size, 1, res, res)
    # Obtain policy model logits
    output = model(recons)
    # Reshape trajectories back into their own dimension
    output = output.view(mask.size(0), channel_size, res)
    # Mask already acquired rows by setting logits to very negative numbers
    loss_mask = (mask == 0).squeeze(-1).squeeze(-2).float()
    logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
    # Also need this for sampling the next row at the end of this loop
    policy = torch.distributions.Categorical(probs)
    return policy, probs


def compute_scores(args, recons, gt_mean, gt_std, unnorm_gt, data_range, comp_psnr=True):
    # For every slice in the batch, and every acquired action per slice, compute the resulting SSIM (and PSNR) scores
    # in parallel.
    # Unnormalise reconstructions
    unnorm_recons = recons * gt_std + gt_mean
    # Reshape targets if necessary (for parallel computation of multiple acquisitions)
    gt_exp = unnorm_gt.expand(-1, recons.shape[1], -1, -1)
    # SSIM scores = batch x k (channels)
    ssim_scores = compute_ssim(unnorm_recons, gt_exp, size_average=False, data_range=data_range).mean(-1).mean(-1)
    # Also compute PSNR
    if comp_psnr:
        psnr_scores = compute_psnr(args, unnorm_recons, gt_exp, data_range)
        return ssim_scores, psnr_scores
    return ssim_scores


def create_data_range_dict(args, loader):
    # Locate ground truths of a volume
    gt_vol_dict = {}
    for it, data in enumerate(loader):
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


def compute_backprop_trajectory(args, kspace, masked_kspace, mask, unnorm_gt, recons, gt_mean, gt_std,
                                data_range, model, recon_model, step, action_list, logprob_list, reward_list):
    # Base score from which to calculate acquisition rewards
    base_score = compute_scores(args, recons, gt_mean, gt_std, unnorm_gt, data_range, comp_psnr=False)
    # Get policy and probabilities.
    policy, probs = get_policy_probs(model, recons, mask)
    # Sample actions from the policy. For greedy (or at step = 0) we sample num_trajectories actions from the
    # current policy. For non-greedy with step > 0, we sample a single action for every of the num_trajectories
    # policies.
    # probs shape = batch x num_traj x res
    # actions shape = batch x num_traj
    # action_logprobs shape = batch x num_traj
    if step == 0 or args.model_type == 'greedy':  # probs has shape batch x 1 x res
        actions = torch.multinomial(probs.squeeze(1), args.num_trajectories, replacement=True)
        actions = actions.unsqueeze(1)  # batch x num_traj -> batch x 1 x num_traj
        # probs shape = batch x 1 x res
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(1)
        actions = actions.squeeze(1)
    else:  # Non-greedy model and step > 0: this means probs has shape batch x num_traj x res
        actions = policy.sample()
        actions = actions.unsqueeze(-1)  # batch x num_traj -> batch x num_traj x 1
        # probs shape = batch x num_traj x res
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(-1)
        actions = actions.squeeze(1)

    # Obtain rewards in parallel by taking actions in parallel
    mask, masked_kspace, zf, recons = compute_next_step_reconstruction(recon_model, kspace,
                                                                       masked_kspace, mask, actions)
    ssim_scores = compute_scores(args, recons, gt_mean, gt_std, unnorm_gt, data_range, comp_psnr=False)
    # batch x num_trajectories
    action_rewards = ssim_scores - base_score
    # batch x 1
    avg_reward = torch.mean(action_rewards, dim=-1, keepdim=True)
    # Store for non-greedy model (we need the full return before we can do a backprop step)
    action_list.append(actions)
    logprob_list.append(action_logprobs)
    reward_list.append(action_rewards)

    if args.model_type == 'greedy':
        # batch x k
        if args.no_baseline:
            # No-baseline
            loss = -1 * (action_logprobs * action_rewards) / actions.size(-1)
        else:
            # Local baseline
            loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(-1) - 1)
        # batch
        loss = loss.sum(dim=1)
        # Average over batch
        # Divide by batches_step to mimic taking mean over larger batch
        loss = loss.mean() / args.batches_step  # For consistency: we generally set batches_step to 1 for greedy
        loss.backward()

        # For greedy: initialise next step by randomly picking one of the measurements for every slice
        # For non-greedy we will continue with the parallel sampled rows stored in masked_kspace, and
        # with mask, zf, and recons.
        idx = random.randint(0, mask.shape[1] - 1)
        mask = mask[:, idx:idx + 1, :, :, :]
        masked_kspace = masked_kspace[:, idx:idx + 1, :, :, :]
        recons = recons[:, idx:idx + 1, :, :]

    elif step != args.acquisition_steps - 1:  # Non-greedy but don't have full return yet.
        loss = torch.zeros(1)  # For logging
    else:  # Final step, can compute non-greedy return
        reward_tensor = torch.stack(reward_list)
        for step, logprobs in enumerate(logprob_list):
            # Discount factor
            gamma_vec = [args.gamma ** (t - step) for t in range(step, args.acquisition_steps)]
            gamma_ten = torch.tensor(gamma_vec).unsqueeze(-1).unsqueeze(-1).to(args.device)
            # step x batch x 1
            avg_rewards_tensor = torch.mean(reward_tensor, dim=2, keepdim=True)
            # Get number of trajectories for correct average
            num_traj = logprobs.size(-1)
            # REINFORCE with self-baselines
            # batch x k
            # TODO: can also store transitions (s, a, r, s') pairs and recompute log probs when
            #  doing gradients? Takes less memory, but more compute: can this be efficiently
            #  batched?
            loss = -1 * (logprobs * torch.sum(
                gamma_ten * (reward_tensor[step:, :, :] - avg_rewards_tensor[step:, :, :]),
                dim=0)) / (num_traj - 1)
            # batch
            loss = loss.sum(dim=1)
            # Average over batch
            # Divide by batches_step to mimic taking mean over larger batch
            loss = loss.mean() / args.batches_step
            loss.backward()  # Store gradients

    return loss, mask, masked_kspace, recons
