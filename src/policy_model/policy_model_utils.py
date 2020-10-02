import torch

from .policy_model_def import build_policy_model

from src.helpers import transforms
from src.helpers.torch_metrics import compute_ssim, compute_psnr


def save_policy_model(args, exp_dir, epoch, model, optimizer, milestones):
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

    if epoch in milestones:
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


def build_optim(args, params):
    optimiser = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimiser


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
    # TODO: Check that this works for greedy model train and eval, and with batch size 1
    if mask.size(1) == mk.size(1) == to_acquire.size(1):
        # Two cases:
        # 1) We are only requesting a single k-space column to acquire per batch.
        # 2) We are requesting multiple k-space columns per batch, and we are already in a trajectory of the non-greedy
        # model: every column in to_acquire corresponds to an existing trajectory that we have sampled the next
        # column for.
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


def compute_next_step_reconstruction(recon_model, kspace, masked_kspace, mask, next_rows):
    # This computation is done by reshaping the masked k-space tensor to (batch . num_trajectories x 1 x res x res)
    # and then reshaping back after performing a reconstruction.
    # TODO: check that this works both for channel size 1 and > 1, als batch_size 1
    mask, masked_kspace = acquire_rows_in_batch_parallel(kspace, masked_kspace, mask, next_rows)
    channel_size = masked_kspace.shape[1]
    # Combine batch and channel dimension for parallel computation if necessary
    res = masked_kspace.size(-2)
    masked_kspace = masked_kspace.view(mask.size(0) * channel_size, 1, res, res, 2)
    zf = get_new_zf(masked_kspace)
    recon = recon_model(zf)
    return mask, masked_kspace, zf, recon


def get_policy_probs(model, recon, mask):
    # Obtain policy model logits
    output = model(recon)
    # Mask already acquired rows by setting logits to very negative numbers
    loss_mask = (mask == 0).squeeze().float()
    logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
    # Also need this for sampling the next row at the end of this loop
    policy = torch.distributions.Categorical(probs)
    return policy, probs


def compute_scores(args, recons, gt_mean, gt_std, unnorm_gt, data_range, comp_psnr=True):
    # For every slice in the batch, and every acquired action per slice, compute the resulting SSIM (and PSNR) scores
    # in parallel.
    # Unnormalise reconstructions
    unnorm_recons = recons * gt_std + gt_mean
    # Reshape targets
    gt_exp = unnorm_gt.expand(-1, args.num_trajectories, -1, -1)
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