import logging
import time
import datetime
import random
import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from src.helpers.utils import (add_mask_params, save_json, check_args_consistency, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters)
from src.helpers.data_loading import create_data_loaders
from src.helpers.metrics import ssim
# Importing Arguments is required for loading of reconstruction model
from src.recon_models.recon_model_utils import load_recon_model, acquire_new_zf, Arguments, acquire_new_zf_exp
from src.impro_models.impro_model_utils import load_impro_model, build_impro_model, build_optim, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)
    for iter, data in enumerate(train_loader):
        kspace, masked_kspace, mask, zf, gt, mean, std, _ = data
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

        loc, logscale = recon_model(zf)
        norm_loc = loc * std + mean  # Back to original scale for metric  # TODO: is this unnormalisation necessary?

        base_score = F.mse_loss(norm_loc, gt, reduction='none').mean(1).mean(1).mean(1)
        # base_score = ssim(norm_loc, gt, size_average=False)  # TODO: Use norm here?

        # Create improvement targets for this batch
        # TODO: Currently this doesn't work, because it loops over the mask for the entire batch simultaneously,
        #  which makes no sense because the mask can be different for every slice. Two solutions:
        #  1) Per slice, vectorise over all rows to acquire, then concatenate these together as a target.
        #  2) Per batch, vectorise over all rows to acquire. This requires checking for every slice whether a
        #     particular row needs acquisition.
        #  Currently we take ~2.5 seconds per slice to generate the full 320 resolution target
        target = torch.zeros((mask.size(0), mask.size(-2))).to(args.device)  # batch_size x resolution

        # Per slice, all rows
        bt = time.perf_counter()
        for sl, (k, mk, m) in enumerate(zip(kspace, masked_kspace, mask)):  # Loop over batch
            st = time.perf_counter()
            # Vectorise masked kspace over all rows to be acquired
            to_acquire = (m[0, 0, :, 0] == 0).nonzero().flatten()
            mk_exp = mk.expand(len(to_acquire), -1, -1, -1).clone()  # TODO: .clone() here?
            # Obtain new zero filled image for all potential rows to acquire
            zf_exp, mean_exp, std_exp = acquire_new_zf_exp(k, mk_exp)
            recon_batch_inds = [k for k in range(0, len(to_acquire), args.krow_batch_size)]
            for j in recon_batch_inds:  # Cannot do ~300 forward passes at the same time, so batch them.
                # Add 'channel' dimension. Treat kspace row dim as batch dim
                batch_zf = zf_exp[j:j + args.krow_batch_size, :, :]
                if batch_zf.size(0) == 0:  # Catch batches of size 0
                    continue
                batch_new_loc, _ = recon_model(batch_zf.unsqueeze(1))
                batch_mean = mean_exp[j:j + args.krow_batch_size].unsqueeze(1)
                batch_std = std_exp[j:j + args.krow_batch_size].unsqueeze(1)
                norm_batch_new_loc = batch_new_loc * batch_std + batch_mean  # TODO: Normalisation necessary?
                batch_gt = gt[sl, ...].expand(norm_batch_new_loc.size(0), -1, -1).unsqueeze(1)
                batch_scores = F.mse_loss(norm_batch_new_loc, batch_gt, reduction='none').mean(1).mean(1).mean(1)
                impros = (base_score[sl] - batch_scores) * 10e11
                target[sl, to_acquire[j:j + args.krow_batch_size]] = impros
            if args.verbose >= 2:
                print('Time to create target for a single slice: {:.3f}s'.format(time.perf_counter() - st))
        if args.verbose >= 1:
            print('Time to create target for batch of size {}: {:.3f}s'.format(gt.size(0), time.perf_counter() - bt))

        # Improvement model output
        output = model(torch.cat((loc, logscale), dim=1), mask.squeeze())

        # Compute loss and backpropagate
        loss = F.l1_loss(output, target)  # TODO: Think about loss function
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, dev_loader, writer):
    # TODO: How to evaluate model succinctly? Maybe skip?
    pass


def visualise(args, epoch, model, display_loader, writer):
    # TODO: What to visualise here?
    pass


def main(args):
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
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimiser = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
        # Create directory to store results in
        args.run_dir = args.exp_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        args.run_dir.mkdir(parents=True, exist_ok=False)

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

    if args.verbose >= 1:
        print('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
            count_parameters(recon_model), count_trainable_parameters(recon_model),
            count_untrainable_parameters(recon_model)))
        print('Improvement model parameters: total {}, of which {} trainable and {} untrainable'.format(
            count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))
    if args.verbose >= 3:
        for p in model.parameters():
            print(p.shape, p.numel())

    # Create data loaders
    train_loader, dev_loader, test_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer)
        # dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        # visualise(args, epoch, model, display_loader, writer)

        # is_new_best = dev_loss < best_dev_loss
        # best_dev_loss = min(best_dev_loss, dev_loss)
        # save_model(args, args.exp_dir, epoch, model, optimiser, best_dev_loss, is_new_best)
        # logging.info(
        #     f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
        #     f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        # )
        save_model(args, args.exp_dir, epoch, model, optimiser, None, False)
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')

    # Data parameters
    parser.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                        help='Which challenge')
    parser.add_argument('--data-path', type=pathlib.Path, required=True,
                        help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str, default=None,
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and None (both used).')

    # Reconstruction model
    parser.add_argument('--recon-model-checkpoint', type=pathlib.Path, required=True,
                        help='Path to a pretrained reconstruction model.')
    parser.add_argument('--use-recon-mask-params', action='store_true',
                        help='Whether to use mask parameter settings (acceleration and center fraction) that the '
                        'reconstruction model was trained on. This will overwrite any other mask settings.')

    # Mask parameters, preferably they match the parameters the reconstruction model was trained on. Also see
    # argument use-recon-mask-params above.
    parser.add_argument('--accelerations', nargs='+', default=[8, 12, 16], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--reciprocals-in-center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')

    parser.add_argument('--num-pools', type=int, default=4, help='Number of ConvNet pooling layers. Note that setting '
                        'this too high will cause size mismatch errors, due to even-odd errors in calculation for '
                        'layer size post-flattening.')
    parser.add_argument('--drop-prob', type=float, default=0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=16, help='Number of ConvNet channels')
    parser.add_argument('--in-chans', default=2, type=int, help='Number of image input channels'
                        'E.g. set to 2 if input is reconstruction and uncertainty map')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--krow-batch-size', default=64, type=int, help='Batch size for target creation loop')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Strength of weight decay regularization')

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

    parser.add_argument('--max-train-slices', type=int, default=None,
                        help='How many slices to train on maximally."')
    parser.add_argument('--max-dev-slices', type=int, default=None,
                        help='How many slices to evaluate on maximally."')
    parser.add_argument('--max-test-slices', type=int, default=None,
                        help='How many slices to test on maximally."')
    parser.add_argument('--verbose', type=str, default=1,
                        help='Set verbosity level. Lowest=0, highest=3."')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
